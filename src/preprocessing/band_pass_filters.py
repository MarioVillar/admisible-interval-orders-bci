from sklearn.base import BaseEstimator, TransformerMixin
from mne.filter import filter_data
import numpy as np


class BandPassFilterEnsemble(BaseEstimator, TransformerMixin):
    """
    Applies several band-pass filters to the data.
    -

    For each frequency range, a band-pass filter is applied to the data.
    A new axis is created in the original data, and the output of each filter is concatenated on it.
    """

    def __validate_params(self) -> None:
        """Validates the parameters of the transformer.

        As described in https://scikit-learn.org/stable/developers/develop.html#instantiation,
        the constructor parameter check cannot be done withint the constructor itself.

        Raises
        ------
        AssertionError
            If the number of frequency ranges is not greater than 0.
            If the number of values in a frequency range is not 2.
            If the lower bound of a frequency range is not lower than the upper bound.
        """
        assert len(self.frec_ranges) > 0, "At least one frequency range must be provided"

        for frec_range in self.frec_ranges:
            assert len(frec_range) == 2, "Each frequency range must have two values"
            assert (
                frec_range[0] < frec_range[1]
            ), "The lower bound must be lower than the upper bound (as it is a band-pass filter)"

    def __init__(self, frec_ranges: list[list[float]] = [], sfreq: float = 1.0) -> None:
        """
        Parameters
        ----------
        frec_ranges : list[list[float]]
            List of frequency ranges to filter.
            Each frequency range must have two values: the lower and upper bound of the range applied
                in the corresponding band-pass filter.
        sfreq
            The sample frequency of the data. By default, 1.0.
        """
        self.frec_ranges = frec_ranges
        self.sfreq = sfreq

    def fit(self, X: np.array, y: np.array = None):
        """
        Only here to make it compatible with sklearn pipelines.
        There are no hyperparameters to tune for this transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ignored.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Ignored.

        Returns
        -------
        self : object
            Fitted (no changes) transformer.
        """
        self.__validate_params()
        return self

    def transform(self, X: np.array, y: np.array = None) -> np.array:
        """
        Applies the band-pass filters to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Ignored

        Returns
        -------
        X_transformed : array-like of shape (len(self.frec_ranges), n_samples, n_features)
            Transformed samples.
            The output of each band-pass filter is concatenated on a new axis.
        """
        self.__validate_params()

        X_transformed = [
            filter_data(X, sfreq=self.sfreq, l_freq=frec_range[0], h_freq=frec_range[1])
            for frec_range in self.frec_ranges
        ]

        return np.array(X_transformed)
