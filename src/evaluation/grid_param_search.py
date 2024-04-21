import ast
from itertools import product
import pandas as pd

import config


def create_param_grid() -> list:
    """
    Parameter grid for SVC, RFC and LDA models.
    """
    param_grid = []

    for svc_param_comb in product(*config.CV_PARAMS_SVC.values()):
        for rfc_param_comb in product(*config.CV_PARAMS_RFC.values()):
            combinacion = [
                dict(zip(config.CV_PARAMS_SVC.keys(), svc_param_comb)),
                dict(zip(config.CV_PARAMS_RFC.keys(), rfc_param_comb)),
                {},
            ]
            param_grid.append(combinacion)

    return param_grid


def get_best_params(results: pd.DataFrame) -> dict:
    """
    Get the optimal parameters according to the mean test score.

    Parameters
    ----------
    results : pd.DataFrame
        Results of the grid search.

    Returns
    -------
    dict
        For each Pipeline, a list of dictionaries with the optimal parameters for each
            model used in the ensemble.
    """
    # Filter out train results
    results = results[results["session"].str.contains("test")]

    # Filter out non useful columns
    results = results[["score", "score_std", "dataset", "pipeline", "param_comb", "freq_bands_ranges"]]

    results = results.groupby(["dataset", "pipeline", "param_comb", "freq_bands_ranges"]).mean().reset_index()

    best_params = {}

    for pipeline in results["pipeline"].unique():
        results_pipeline = results[results["pipeline"] == pipeline]

        best_results_pipeline = results_pipeline.iloc[results_pipeline["score"].argmax()]

        best_params[pipeline] = {
            "param_comb": ast.literal_eval(best_results_pipeline["param_comb"]),
            "freq_bands_ranges": ast.literal_eval(best_results_pipeline["freq_bands_ranges"]),
        }

    return best_params
