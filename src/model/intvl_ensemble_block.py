from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any
from abc import ABC, abstractmethod

from .intvl_model_ensemble import IntvlModelEnsemble


class IntvlEnsembleBlock(ABC, BaseEstimator, ClassifierMixin):
    """
    Block of model ensembles for each frecuency band range.
    -

    A IntvlModelEnsemble is used on each frecuency band range. The model types comprising each individual
    ensemble are the same. Then, the predictions of each frequency range and class integration
    is should be defined by the child classes.
    """

    def __validate_params(self):
        assert len(self.ind_ensembles) > 0, "There should be at least one individual ensemble"
        assert 0 <= self.alpha <= 1, "Alpha should be between 0 and 1"

        self._estimator_type = "regressor"
        self.n_frec_ranges = len(self.ind_ensembles)
        self.fuzzy_measure = self.card_fuzzy_measure()

        self.validated_params = True

    def __init__(
        self,
        ind_ensembles: list = [],
        alpha: float = 1,
        p: float = 2,
        n_jobs: int = -1,
        _estimator_type: Any = None,
        fuzzy_measure: Any = None,
        n_frec_ranges: int = -1,
        validated_params: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        - ind_ensembles : list
            List of IntvlModelEnsemble objects.
        - alpha : float
            The weight used in the K-alpha mapping of the intervals in order to compare them. The child class
            may use it for interval ordering.
        - p : float
            Exponent to which the expression of the fuzzy_measure is raised. The child class may use it for
            the fuzzy measure employed.
        - n_jobs : int
            Number of jobs to run in parallel.
        - _estimator_type : str
            Not used. Only for scikit-learn compatibility purposes (so it gest added to get_params() method).
        - fuzzy_measure : array-like of shape (N,)
            Not used. Only for scikit-learn compatibility purposes (so it gest added to get_params() method).
        - n_frec_ranges : int
            Number of frecuency band ranges. Used to determine the number of individual ensembles created.
                Not used. Only for scikit-learn compatibility purposes (so it gest added to get_params() method).
        - validated_params : bool
            Whether the parameters have been validated or not. If not, they are validated before fittingNot used.
                Only for scikit-learn compatibility purposes (so it gest added to get_params() method).
        """
        self.ind_ensembles = ind_ensembles
        self.alpha = alpha
        self.p = p
        self.n_jobs = n_jobs
        self._estimator_type = _estimator_type
        self.fuzzy_measure = fuzzy_measure
        self.n_frec_ranges = n_frec_ranges
        self.validated_params = validated_params

    @classmethod
    def create_ensemble(
        cls,
        model_class_list: list,
        model_class_names: list,
        n_frec_ranges: int,
        model_class_kwargs: list = None,
        ind_model_ens_kwargs: dict = {},
        p: float = 2,
        alpha: float = 1,
        n_jobs: int = -1,
    ):
        """
        Create a block of ensembles of IntvlModelEnsemble objects.

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
        - p : float
            Exponent to which the expression of the fuzzy_measure is raised.
        - alpha : float
            The weight used in the K-alpha mapping of the intervals in order to compare them.
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

        return cls(ind_ensembles=ind_ensembles, p=p, alpha=alpha, n_jobs=n_jobs)

    def card_fuzzy_measure(self) -> np.ndarray:
        """
        Fuzzy measure based on the cardinality of a set.
        -
        Calculate the cardinality fuzzy measure of each subset of a set with size N.

        Returns
        -------
        - fuzzy_measure : array-like of shape (n_frec_ranges+1,)
            Value of the fuzzy measure of each subset of the set (including the empty set).
        """
        return np.array([(x / self.n_frec_ranges) ** self.p for x in np.arange(0, self.n_frec_ranges + 1)])

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
        # Obtain the class probabilities predictions
        class_probs = self.predict_proba(X)

        # Return the class with the highest probability
        return np.argmax(class_probs, axis=1)

    @abstractmethod
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
        pass

    def _predict_ensembles_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities for X for each model ensemble that comprises the block.

        Parameters
        ----------
        - X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        Returns
        -------
        - y : array-like of shape (n_samples, n_classes, n_frec_ranges, 2)
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

        return y
