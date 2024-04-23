import numpy as np
import ast
import pandas as pd


def load_eegbci_time_intvls_results(file_path: str) -> pd.DataFrame:
    """
    Load the EEGBCI time intervals results from the csv file.

    Parameters
    ----------
    file_path : str
        The file path to the EEGBCI time intervals results csv file.

    Returns
    -------
    pd.DataFrame
        The EEGBCI time intervals results.
    """
    results = pd.read_csv(file_path)

    results["scores"] = results["scores"].apply(ast.literal_eval)
    results["w_times"] = results["w_times"].apply(
        lambda x: np.array(x.replace("[", "").replace("]", "").replace("\n", "").split(), dtype=np.float_)
    )
    results["param_comb"] = results["param_comb"].apply(ast.literal_eval)
    results["freq_bands_ranges"] = results["freq_bands_ranges"].apply(ast.literal_eval)

    return results
