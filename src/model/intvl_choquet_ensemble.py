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
        # The output is reshaped to the original shape but the dimension of each model (which
        #   each agreggated by the choquet integral)
        y = np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(self.intvl_choquet_integ)(intvl_set) for intvl_set in y_flattened),
        ).reshape((y.shape[0], y.shape[1], y.shape[3]))

        # Apply the K-alpha and K-beta mappings to each interval prediction. It multiplies the infimum by
        #   1-alpha or 1-beta and the supremum by alpha or beta.
        alpha_mapping = np.dot(y[:, :, :], [1 - self.alpha, self.alpha])
        beta_mapping = np.dot(y[:, :, :], [1 - self.beta, self.beta])

        alpha_beta_mapping = np.stack((alpha_mapping, beta_mapping), axis=2)

        return alpha_beta_mapping

    def predict(self, X: np.array) -> np.array:
        """
        Predict class labels for X.

        For each sample, the winning class is the higher ranked in the order
        determined by the K-alpha and K-beta mappings.

        Parameters
        ----------
        - X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        Returns
        -------
        - y : array-like of shape (n_samples,)
            Predicted class label per sample.
        """
        # Obtain the class probabilities predictions
        class_probs = self.predict_proba(X)  # Shape (n_samples, n_classes, 2) -> alpha and beta mappings

        # For each sample, return the class with the highest K-alpha,K-beta mapping
        def k_alpha_beta_argmax(sample):
            max_index = 0
            max_value = sample[0]  # Contains the K-alpha and K-beta mappings

            for i in range(1, len(sample)):
                if (sample[i][0] > max_value[0]) or (sample[i][0] == max_value[0] and sample[i][1] > max_value[1]):
                    max_index = i
                    max_value = sample[i]

            return max_index

        # Return the winning class for each of the samples received
        return np.apply_along_axis(k_alpha_beta_argmax, axis=1, arr=class_probs)

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

        # Defined for parallelization purposes
        def compute_choquet_integ_permu(sigma):
            """
            Compute the interval Choquet integral for a specific admissible permutation.

            Parameters
            ----------
            - sigma : array-like of shape (n_frec_ranges,)
                Current permutation of the set of intervals. Index i contains the index in the original vector of
                    the element that has been mapped to position i by the permutation.

            Returns
            -------
            - choquet_integ_permu : list of shape (2,2)
                First component has the meaning of whether the permutation was admissible (1) or not (0). It has
                    shape (2,) to be able to use np arrays over the Parallel function result. The sum of its first
                    component is the number of admissible permutations found.
                Second component is the resulting interval obtained by the Choquet integral (admissible permutations)
                    or zero interval (non-admissible permutations, so they do not influence on the summatory).
            """
            if self.is_admissible_permu(intvl_set=intvl_set, sigma=sigma):
                return [[1, 1], self.intvl_choquet_integ_permu(intvl_set, np.array(sigma))]
            else:
                return [[0, 0], [0, 0]]

        orig_indexes = np.arange(self.n_frec_ranges)

        # Obtain all permutations
        sigma_list = permutations(orig_indexes)

        # Run in parallel mode
        results = np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(compute_choquet_integ_permu)(sigma) for sigma in sigma_list)
        )

        # Compute the sum of interval Choquet integrals and count the number of admissible permutations
        n_admissible_permus = np.sum(results[:, 0, 0])
        choquet_integ_sum = np.sum(results[:, 1], axis=0)

        return choquet_integ_sum / n_admissible_permus

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
        ele_check_mask = (intvl_set[first_ele_idx][:, 0] < intvl_set[second_ele_idx][:, 0]) | (
            (intvl_set[first_ele_idx][:, 0] == intvl_set[second_ele_idx][:, 0])
            & (intvl_set[first_ele_idx][:, 1] < intvl_set[second_ele_idx][:, 1])
        )

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
