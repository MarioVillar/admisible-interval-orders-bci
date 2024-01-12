import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.utils.validation import check_is_fitted


class IntvlModelEnsemble(VotingClassifier):
    """Interval Choquet integral classification model ensemble.

    The predictions of each class is returned as an interval. For each class, the infimum and supremum of the
        corresponding interval are derived by determining the minimum and maximum predictions, respectively,
        among the models comprising the ensemble.
    To determine the winning class, the interval Choquet integral is applied to the interval predictions.
    """

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_frec_ranges, n_samples, n_csp_features)
            Input samples.

        Returns
        -------
        p : array-like of shape (n_classes, 2)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        check_is_fitted(self)

        proba_list = np.asarray([clf.predict_proba(X) for clf in self.estimators_])

        return np.array([np.min(proba_list, axis=0), np.max(proba_list, axis=0)])
