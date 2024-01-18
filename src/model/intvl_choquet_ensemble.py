from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any
from itertools import permutations

from .intvl_model_ensemble import IntvlModelEnsemble


class IntvlChoquetEnsemble(BaseEstimator, ClassifierMixin):
    """
    Model ensemble of each frecuency band range.
    -

    A IntvlModelEnsemble is used on each frecuency band range. The model types comprising each individual
    ensemble are the same.
    """

    def __validate_params(self):
        assert len(self.ind_ensembles) > 0, "There should be at least one individual ensemble"

        self._estimator_type = "regressor"
        self.fuzzy_measure = self.card_fuzzy_measure(len(self.ind_ensembles))

    def __init__(
        self, ind_ensembles: list = [], n_jobs: int = -1, _estimator_type: Any = None, fuzzy_measure: Any = None
    ) -> None:
        """
        Parameters
        ----------
        - ind_ensembles : list
            List of IntvlModelEnsemble objects.
        - n_jobs : int
            Number of jobs to run in parallel.
        - _estimator_type : str
            Not used. Only for scikit-learn compatibility purposes (so it gest added to get_params() method).
        - fuzzy_measure : array-like of shape (N,)
            Not used. Only for scikit-learn compatibility purposes (so it gest added to get_params() method).
        """
        self.ind_ensembles = ind_ensembles
        self.n_jobs = n_jobs
        self._estimator_type = _estimator_type
        self.fuzzy_measure = fuzzy_measure

    @classmethod
    def create_ensemble(
        cls,
        model_class_list: list,
        model_class_names: list,
        n_frec_ranges: int,
        model_class_kwargs: list = None,
        ind_model_ens_kwargs: dict = {},
        n_jobs: int = -1,
    ):
        """
        Create an ensemble of IntvlModelEnsemble objects.

        Parameters
        ----------
        - model_class_list : list
            List of model types classes to be used in each individual ensemble. They should be scikit-learn-classifier-typed.
        - model_class_names : list
            List of model types names to be used when creating each model within the individual ensembles.
        - n_frec_ranges : int
            Number of frecuency band ranges. Used to determine the number of individual ensembles created.
        - model_class_kwargs : list
            List of dictionaries containing the key arguments for each model type class.
        - ind_model_ens_kwargs : dict
            Key arguments for each individual ensemble.
        - n_jobs : int
            Number of jobs to run in parallel.

        Returns
        -------
        - ensemble : object of IntvlChoquetEnsemble
            The created ensemble.
        """
        assert len(model_class_list) > 0, "There should be at least one model class"
        assert n_frec_ranges > 0, "There should be at least one frecuency band range"
        assert len(model_class_names) == len(model_class_names), "There should be one name per model type class"

        if model_class_kwargs is not None:
            assert len(model_class_list) == len(
                model_class_kwargs
            ), "There should be one set of arguments per model class. If no arguments are needed, an empty dictionary should be passed."
        else:
            model_class_kwargs = [{} for _ in range(len(model_class_list))]

        ind_ensembles = []

        # For each frecuency band range, an individual ensemble should be created
        for i in range(n_frec_ranges):
            ind_estimators = []

            # All individual ensembles are formed by the same model types.
            # For each model type, an instance is created (with its corresponding key arguments) and added to the ensemble
            for j in range(len(model_class_list)):
                ind_estimators.append((f"{model_class_names[j]}_{i}", model_class_list[j](**model_class_kwargs[j])))

            # Create and store a new individual ensemble
            ind_ensembles.append(IntvlModelEnsemble(estimators=ind_estimators, **ind_model_ens_kwargs))

        return cls(ind_ensembles=ind_ensembles, n_jobs=n_jobs)

    def fit(self, X: np.array, y: np.array = None):
        """
        Fit each individual ensemble.

        Parameters
        ----------
        - X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        - y :  array-like of shape (n_samples,) or (n_samples, n_outputs)
            The class for each epoch.

        Returns
        -------
        - self : object of IntvlChoquetEnsemble
            The fitted instance.
        """
        self.__validate_params()

        # Defined for parallelization purposes
        def fit_ind_ensemble(x: np.array, ensemble: IntvlModelEnsemble):
            return ensemble.fit(x, y)

        # Run in parallel mode
        #   For each individual ensemble, the corresponding frecuency band range date is passed and it is fitted on it
        #   The fitted individual ensembles are returned and overwrite the previous ones
        self.ind_ensembles = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_ind_ensemble)(x, ensemble) for x, ensemble in zip(X, self.ind_ensembles)
        )

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict class labels for X.

        Parameters
        ----------
        - X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        Returns
        -------
        - y : array-like of shape (n_samples,)
            Predicted class label per sample.
        """
        class_probs = self.predict_proba(X)

        # This should be done by using a K-lambda mapping
        return np.argmax(class_probs, axis=1)

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        - X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        Returns
        -------
        - y : array-like of shape (n_frec_ranges, n_classes, n_samples, 2)
            Class probabilities of the input samples.
        """
        self.__validate_params()

        # Defined for parallelization purposes
        def predict_proba_ind_ensemble(x: np.array, ensemble: IntvlModelEnsemble):
            return ensemble.predict_proba(x)

        # Run in parallel mode
        # Return shape: (n_samples, n_classes, n_frec_ranges, 2)
        y = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_ind_ensemble)(x, ensemble) for x, ensemble in zip(X, self.ind_ensembles)
            ),
            axis=2,
        )

        # Flatten the array in order to apply the interval Choquet integral to each sample and each class prediction
        y_flattened = y.reshape((-1, y.shape[2], y.shape[3]))

        # Run in parallel mode
        # The output is reshaped to the original shape but the dimension of each model (which each agreggated by the choquet integral)
        y = np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(self.intvl_choquet_integ)(intvl_set) for intvl_set in y_flattened),
        ).reshape((y.shape[0], y.shape[1], y.shape[3]))

        return y

    def card_fuzzy_measure(self, N: int, p: float = 1) -> np.ndarray:
        """
        Fuzzy measure based on the cardinality of a set.
        -
        Calculate the cardinality fuzzy measure of each subset of a set with size N.

        Parameters
        ----------
        - N : int
            Cardinality of the set.
        - p : float
            Exponent to which the expression is raised.

        Returns
        -------
        - fuzzy_measure : array-like of shape (N+1,)
            Value of the fuzzy measure of each subset of the set (including the empty set).
        """
        return np.array([(x / N) ** p for x in np.arange(0, N + 1)])

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

        print("Another one")

        orig_indexes = np.arange(len(intvl_set))

        # Obtain all permutations
        sigma_list = permutations(orig_indexes)

        # Run in parallel mode
        results = np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(compute_choquet_integ_permu)(sigma) for sigma in sigma_list)
        )

        # Compute the sum of interval Choquet integrals and count the number of admissible permutations
        n_admissible_permus = np.sum(results[:, 0, 0])
        choquet_integ_sum = np.sum(results[:, 1], axis=0)

        # for sigma in sigma_list:
        #     # Check if the permutation is admissible
        #     if self.is_admissible_permu(intvl_set=intvl_set, sigma=sigma):
        #         # If so, compute the interval Choquet integral for that permutation
        #         choquet_integ_sum += self.intvl_choquet_integ_permu(intvl_set, np.array(sigma))
        #         n_admissible_permus += 1

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

        n = len(intvl_set)  # n_frec_ranges TODO: Create an instance attribute for n_frec_ranges
        sigma_inv = np.argsort(sigma)

        ###############################
        # First condition
        # Obtain all the index pairs of the upper-triangle and lower-triangle of an n x n matrix.
        # k=1 or k=-1 allows to exclude the diagonal of the matrix (there is no need to check each elemento with itself).
        # These are all the pair of indexes that should be check for the first condition of admissibility.
        i, j = np.triu_indices(n, k=1)  # Start from 1 diagonal above the main diagonal
        h, t = np.tril_indices(n, k=-1)  # Start from 1 diagonal below the main diagonal

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

            # ###############################
            # # Second condition (alternative using numpy - it is actually slower)
            # # Boolean mask of dim n_frec_ranges x n_frec_ranges
            # # The element (i,j) is True if the interval i of the invtl_set is equal to the interval j
            # logic_mat = (intvl_set[:, 0, np.newaxis] == intvl_set[:, 0]) & (
            #     intvl_set[:, 1, np.newaxis] == intvl_set[:, 1]
            # )
            # # logic_mat = np.logical_and(
            # #     np.expand_dims(intvl_set[:, 0], axis=1) == intvl_set[:, 0],
            # #     np.expand_dims(intvl_set[:, 1], axis=1) == intvl_set[:, 1],
            # # )

            # # Map the sigma permutation to each of the columns of the logic matrix. If the the permutation is
            # #   admissible, the True values of each column should be consecutive.
            # logic_mat_sigma = logic_mat[sigma_inv, :]

            # # Get the difference between consecutive rows of each column of the logic matrix
            # diff = np.diff(logic_mat_sigma.astype(np.int8), axis=0)

            # # Find the index of the first occurrence of -1
            # index_of_minus_1 = np.argmax(np.vstack((diff, -np.ones(diff.shape[1], dtype=np.int8))) == -1, axis=0)
            # # index_of_minus_1 = np.argmax(diff == -1, axis=0)

            # # Generate a matrix with each row index in each column
            # row_index_matrix = np.zeros(diff.shape) + np.arange(diff.shape[0], dtype=np.int32)[:, np.newaxis]

            # # Get the elements of the matrix whose row index in the corresponding column is greater
            # #   than the index of the first -1. Check if there are any 1s in that subset.
            # is_admissible = not np.any(diff[row_index_matrix > index_of_minus_1] == 1)

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
        n = len(intvl_set)  # n_frec_ranges

        # Obtain the array of fuzzy measure differences that multiplies each interval in the summatory
        fuzzy_measure_diff = (
            self.fuzzy_measure[n - np.arange(len(sigma))] - self.fuzzy_measure[n - np.arange(len(sigma)) - 1]
        )

        # Multiply each fuzzy measure difference by the corresponding interval and compute the sumatory of resulting intervals
        return np.sum((intvl_set[sigma].transpose() * fuzzy_measure_diff).transpose(), axis=0)
