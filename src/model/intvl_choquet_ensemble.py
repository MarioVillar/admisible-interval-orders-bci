from joblib import Parallel, delayed
import numpy as np
from itertools import permutations

from .intvl_ensemble_block import IntvlEnsembleBlock


class IntvlChoquetEnsemble(IntvlEnsembleBlock):
    """
    Block of model ensembles for each frecuency band range, where the model ensembles
    aggregation is done by the Choquet integral.
    -

    A IntvlModelEnsemble is used on each frecuency band range. The model types comprising each individual
    ensemble are the same. Then, the predictions of each frequency range and class are integrated in the
    following way:
    1. For each class, obtain the interval Choquet integral of the predicted intervals of each of the
        frecuency band ranges.
    2. For each class, compute the K-alpha mapping of the interval obtained in the previous step.
    3. The class with the highest K-alpha value is the winning class.
    """

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        - X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        Returns
        -------
        - y : array-like of shape (n_samples, n_classes, 2)
            Class probabilities of the input samples.
        """
        y = self._predict_ensembles_proba(X=X)

        # Flatten the array in order to apply the interval Choquet integral to each sample and each class prediction
        y_flattened = y.reshape((-1, y.shape[2], y.shape[3]))

        # Run in parallel mode
        # The output is reshaped to the original shape but the dimension of each model (which each agreggated by the choquet integral)
        y = np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(self.intvl_choquet_integ)(intvl_set) for intvl_set in y_flattened),
        ).reshape((y.shape[0], y.shape[1], y.shape[3]))

        # Apply the K-alpha mapping to each interval prediction. It multiplies the infimum by 1-alpha and the supremum by alpha.
        alpha_mapping = np.dot(y[:, :, :], [1 - self.alpha, self.alpha])

        return alpha_mapping

    def intvl_choquet_integ(self, intvl_set: np.ndarray) -> np.ndarray:
        """
        Interval Choquet integral.
        -
        The resulting value is obtained as the arithmetic mean of the interval Choquet integrals
        computed for each admissible permutation.

        A permutation sigma is admissible if:
        - for every` x_i < x_j` , we have that `sigma^-1(i) < sigma^-1(j)`
        - for each `x_i`, the set `{sigma^-1(j) | j ∈ {1,...,n} with x_i = x_j }` is an interval in `N` (natutal
            numbers set).

        Where `sigma^-1` is the inverse permutation, i.e., the function that maps each original index to the permutated one.

        Parameters
        ----------
        - intvl_set : array-like of shape (n_frec_ranges, 2)
            Set of intervals to be used in the Choquet integral.

        Returns
        -------
        - intvl_choquet_integ : array-like of shape (2,)
            Resulting interval obtained by the Choquet integral.
        """
        orig_indexes = np.arange(self.n_frec_ranges)

        # Obtain all permutations
        sigma_list = permutations(orig_indexes)

        # Filter just the admissible permutations
        sigma_list = [sigma for sigma in sigma_list if self.is_admissible_permu(intvl_set=intvl_set, sigma=sigma)]

        # If not all the admissible permutations should be used
        if self.choquet_n_permu != -1:
            # Compute the degrees of totalness for each permutation. Shape -> (len(sigma_list), n_frec_ranges-1)
            dt_permus = np.array([self.degree_totalness(intvl_set=intvl_set, sigma=np.array(i)) for i in sigma_list])

            # Compute M1 and M2 aggregation functions (average and minimum)
            mean_dt_permus = np.mean(dt_permus, axis=1)
            min_dt_permus = np.min(dt_permus, axis=1)

            # Sort first by M1 and then by M2
            idx_permu = np.lexsort((min_dt_permus, mean_dt_permus))

            # Select the `self.choquet_n_permu` best permutations
            idx_best_permu = idx_permu[: self.choquet_n_permu]

            # Update the list of permutations to be used
            sigma_list = [sigma_list[i] for i in idx_best_permu]

        # Run in parallel mode
        results = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self.intvl_choquet_integ_permu)(intvl_set, np.array(sigma)) for sigma in sigma_list
            )
        )

        # Compute the sum of interval Choquet integrals
        choquet_integ_sum = np.sum(results, axis=0)

        return choquet_integ_sum / len(sigma_list)

    def is_admissible_permu(self, intvl_set: np.ndarray, sigma: np.ndarray) -> bool:
        """
        Check if a permutation is admissible.
        -
        A permutation sigma is admissible if:
        - for every `x_i < x_j | x_i,x_j ∈ intvl_set` , we have that `sigma^-1(i) < sigma^-1(j)`
        - for each `x_i`, the set `{sigma^-1(j) | j ∈ {1,...,n} with x_i = x_j }` is an interval in `N` (natutal
            numbers set).

        Where `sigma^-1` is the inverse permutation, i.e., the function that maps each original index to the permutated one.

        The order used between intervals is ass follows:
            `x <= y if and only if x[0] <= y[0] and x[1] <= y[1]`
        Which leads to:
            `x < y if and only if x[0] < y[0] or x[0] == y[0] and x[1] < y[1]`

            `x > y if and only if x[0] > y[0] or x[0] == y[0] and x[1] > y[1]`

            `x == y if and only if x[0] == y[0] and x[1] == y[1]`

        Important:
        -
        This implementation fails when `n_frec_ranges > 20000`

        Parameters
        ----------
        - intvl_set : array-like of shape (n_frec_ranges, 2)
            Set of intervals to be used in the Choquet integral.
        - sigma : array-like of shape (n_frec_ranges,)
            Current permutation of the set of intervals. Index i contains the index in the original vector of
            the element that has been mapped to position i by the permutation.

        Returns
        -------
        - is_admissible : bool
            Whether the permutation is admissible or not
        """
        is_admissible = True

        sigma_inv = np.argsort(sigma)

        ###############################
        # First condition
        # Obtain all the index pairs of the upper-triangle and lower-triangle of an n x n matrix.
        # k=1 or k=-1 allows to exclude the diagonal of the matrix (there is no need to check each element with itself).
        # These are all the pair of indexes that should be check for the first condition of admissibility.
        i, j = np.triu_indices(self.n_frec_ranges, k=1)  # Start from 1 diagonal above the main diagonal
        h, t = np.tril_indices(self.n_frec_ranges, k=-1)  # Start from 1 diagonal below the main diagonal

        first_ele_idx = np.concatenate((i, h))
        second_ele_idx = np.concatenate((j, t))

        # Get the pairs of intervals in which the first interval is smaller than the second one.
        # Check the method docummentation for further information on how intervals are compared.
        ele_check_mask = (intvl_set[first_ele_idx][:, 0] < intvl_set[second_ele_idx][:, 0]) & (
            intvl_set[first_ele_idx][:, 1] < intvl_set[second_ele_idx][:, 1]
        )

        # This following mask would be using the lexicographical order with respect to the first component.
        # ele_check_mask = (intvl_set[first_ele_idx][:, 0] < intvl_set[second_ele_idx][:, 0]) | (
        #     (intvl_set[first_ele_idx][:, 0] == intvl_set[second_ele_idx][:, 0])
        #     & (intvl_set[first_ele_idx][:, 1] < intvl_set[second_ele_idx][:, 1])
        # )

        # Get the actual indexes that have to be checked.
        first_ele_idx = first_ele_idx[ele_check_mask]
        second_ele_idx = second_ele_idx[ele_check_mask]

        # Check if the condition for the inverse of the permutation holds for all the previous elements
        is_admissible = np.all(sigma_inv[first_ele_idx] < sigma_inv[second_ele_idx])

        ###############################
        # Second condition
        if is_admissible:
            # For each interval, get the permutation inverse of the indexes of the intervals that are equal to it
            for intvl in intvl_set:
                equal_list = sigma_inv[
                    np.where(np.logical_and(intvl_set[:, 0] == intvl[0], intvl_set[:, 1] == intvl[1]))[0]
                ]

                # It is only admissible if all the elements are consecutive
                is_admissible = np.all(np.diff(equal_list) == 1)

                if not is_admissible:
                    break

        return is_admissible

    def intvl_choquet_integ_permu(self, intvl_set: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Interval Choquet integral given a specific permutation.

        Parameters
        ----------
        - intvl_set : array-like of shape (n_frec_ranges,2)
            Set of intervals to be used in the Choquet integral.
        - sigma : array-like of shape (n_frec_ranges,)
            Current permutation of the set of intervals. Index i contains the index in the original vector of
            the element that has been mapped to position i by the permutation.

        Returns
        -------
        - intvl_choquet_integ : array-like of shape (2,)
            Resulting interval obtained by the Choquet integral.
        """
        # Obtain the array of fuzzy measure differences that multiplies each interval in the summatory
        fuzzy_measure_diff = (
            self.fuzzy_measure[self.n_frec_ranges - np.arange(len(sigma))]
            - self.fuzzy_measure[self.n_frec_ranges - np.arange(len(sigma)) - 1]
        )

        # Multiply each fuzzy measure difference by the corresponding interval and compute the sumatory of resulting intervals
        return np.sum((intvl_set[sigma].transpose() * fuzzy_measure_diff).transpose(), axis=0)

    def degree_totalness(self, intvl_set: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Totalness degree of a permutation.

        Parameters
        ----------
        - intvl_set : array-like of shape (n_frec_ranges,2)
            Set of intervals to be used in the Choquet integral.
        - sigma : array-like of shape (n_frec_ranges,)
            Current permutation of the set of intervals. Index i contains the index in the original vector of
            the element that has been mapped to position i by the permutation.

        Returns
        -------
        - dt : array-like of shape (len(sigma)-1,)
            Degree of totalness of the permutation.
        """
        dt = np.zeros(len(sigma) - 1)

        for i in range(0, len(sigma) - 1):
            dt[i] = 1 - max(
                0,
                intvl_set[sigma[i]][0] - intvl_set[sigma[i + 1]][0],
                intvl_set[sigma[i]][1] - intvl_set[sigma[i + 1]][1],
            )

        return dt
