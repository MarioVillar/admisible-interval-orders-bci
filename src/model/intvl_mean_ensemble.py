import numpy as np

from .intvl_ensemble_block import IntvlEnsembleBlock


class IntvlMeanEnsemble(IntvlEnsembleBlock):
    """
    Mean interval model ensemble of each frecuency band range.
    -

    A IntvlModelEnsemble is used on each frecuency band range. The model types comprising each individual
    ensemble are the same. Then, the predictions of each frequency range and class are integrated in the
    following way:
    1. For each frequency band range and each class, obtain the mean of the predicted interval, i.e., for
        [a, b] the mean is (a+b)/2.
    2. For each class, compute the mean of the mean values obtained in the previous step.
    3. The class with the highest mean value is the winning class.
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
        - y : array-like of shape (n_samples, n_classes)
            Class probabilities of the input samples.
        """
        y = self._predict_ensembles_proba(X=X)

        # Get mean value of each interval
        y = np.mean(y, axis=3)

        # Get mean value of each frequency band range
        y = np.mean(y, axis=2)

        return y
