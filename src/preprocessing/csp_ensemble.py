from mne.decoding import CSP
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

from utils import parallelization


class CSPEnsemble(BaseEstimator, TransformerMixin):
    """
    CSP Ensemble applied to each frequency range.
    -

    Applies CPS to each of the data series separated by band-pass filters in a previous step.
    """

    def _validate_params(self, check_csp_list: bool = False) -> None:
        """Validates the parameters of the transformer.

        As described in https://scikit-learn.org/stable/developers/develop.html#instantiation,
        the constructor parameter check cannot be done withint the constructor itself.

        Parameters
        ----------
        check_csp_list : bool
            Whether to check if all the CSP objects have been added to the list or not.

        Raises
        ------
        AssertionError
            If the number of components to keep is not greater than 0.
            If the number of frequency ranges is not greater than 0.
        """
        assert self.n_components > 0, "At least one component should be kept when applying CSP"
        assert self.n_frec_ranges > 0, "At least there should be one frequency range"

        if check_csp_list:
            assert len(self.csp_list) == self.n_frec_ranges, "There should be one CSP object per frequency range"

    def __init__(self, n_components: int = 4, n_frec_ranges: int = -1, n_jobs: int = -1, **kwargs) -> None:
        """
        Parameters
        ----------
        n_components : int
            Number of components to keep while applying CSP.
        n_frec_ranges : int
            Number of frequency ranges in which data has been filtered.
        n_jobs : int
            Number of jobs to run in parallel.

        Other parameters
        ----------------
        kwargs
            Other parameters to be passed to the CSP constructor.
        """
        self.n_components = n_components
        self.n_frec_ranges = n_frec_ranges
        self.csp_list = []  # List of CSP objects, one for each frequency range
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        X : array-like of shape (n_frec_ranges, n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs)
            The class for each epoch.

        Returns
        -------
        self : object of CSPEnsemble
            The fitted instance.
        """
        self._validate_params()

        # Defined for parallelization purposes
        def fit_csp(series):
            # return CSP(n_components=self.n_components, **self.kwargs).fit(series, y)
            return CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False, **self.kwargs).fit(
                series, y
            )

        # Run in parallel mode
        orig_set = parallelization.get_current_childs()
        self.csp_list = Parallel(n_jobs=self.n_jobs)(delayed(fit_csp)(series) for series in X)
        parallelization.kill_diff_childs(orig_set)

        # # For each frequency range, create a new CSP object, fit it to the data
        # #   and store it in the list of CSP objects
        # for series in X:
        #     print("Series shape: ", series.shape)
        #     csp = CSP(n_components=self.n_components, **self.kwargs)
        #     csp.fit(series, y)
        #     self.csp_list.append(csp)

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """
        Applies CSP to each of the series in which the data has been divided by the band-pass filters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Ignored

        Returns
        -------
        X_transformed : array-like of shape (n_frec_ranges, n_samples, n_csp_features)
            Transformed samples.
            The output of each band-pass filter is concatenated on a new axis.
        """
        self._validate_params(check_csp_list=True)

        # Defined for parallelization purposes
        def transform_csp(series, csp):
            return csp.transform(series)

        # Run in parallel mode
        orig_set = parallelization.get_current_childs()
        X_transformed = Parallel(n_jobs=self.n_jobs)(
            delayed(transform_csp)(series, csp) for series, csp in zip(X, self.csp_list)
        )
        parallelization.kill_diff_childs(orig_set)

        return np.array(X_transformed)
