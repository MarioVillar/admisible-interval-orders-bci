from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any

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
        self._estimator_type = "regressor"
        self.fuzzy_measure = self.card_fuzzy_measure(len(self.ind_ensembles))

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
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_proba_ind_ensemble)(x, ensemble) for x, ensemble in zip(X, self.ind_ensembles)
        )

        # This should be done by Interval Choquet Integral
        y = np.mean(np.array(y), axis=1)
        y = np.mean(np.array(y), axis=0)

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

    def intvl_choquet_integ_permu(self, intvl_set: np.ndarray, permu_idxs: np.ndarray) -> np.ndarray:
        """
        Interval Choquet integral given a specific permutation.

        Parameters
        ----------
        - intvl_set : array-like of shape (n_frec_ranges,)
            Set of intervals to be used in the Choquet integral.
        - permu_idxs : array-like of shape (n_frec_ranges,)
            Current permutation of the set of intervals.

        Returns
        -------
        - intvl_choquet_integ : array-like of shape (2,)
            Resulting interval obtained by the Choquet integral.
        """
        n = len(intvl_set)  # n_frec_ranges

        # Obtain the array of fuzzy measure differences that multiplies each interval in the summatory
        fuzzy_measure_diff = (
            self.fuzzy_measure[n - np.arange(len(permu_idxs))] - self.fuzzy_measure[n - np.arange(len(permu_idxs)) - 1]
        )

        # Multiply each fuzzy measure difference by the corresponding interval and compute the sumatory of resulting intervals
        return np.sum((intvl_set[permu_idxs].transpose() * fuzzy_measure_diff).transpose(), axis=0)
