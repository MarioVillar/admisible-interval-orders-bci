import os
import warnings
import mne
import pandas as pd
from sklearn.pipeline import make_pipeline
import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

import config
from utils.disk import save_results_csv

from evaluation.grid_param_search import create_param_grid

from preprocessing.band_pass_filters import BandPassFilterEnsemble
from preprocessing.csp_ensemble import CSPEnsemble

from model.intvl_choquet_ensemble import IntvlChoquetEnsemble
from model.intvl_mean_ensemble import IntvlMeanEnsemble
from model.intvl_sugeno_ensemble import IntvlSugenoEnsemble


mne.set_log_level("CRITICAL")
moabb.set_log_level("ERROR")
warnings.filterwarnings("ignore")


##############################################################################
# Initializing Datasets

dataset = BNCI2014_001()
# dataset.subject_list = [1]

# Obtain sample frequency of the data
subject_data = dataset.get_data(subjects=[1])
sfreq = subject_data[1][list(subject_data[1].keys())[0]]["0"].info["sfreq"]

#############################################################################
# Create param grid and initialize results dataframe

param_grid = create_param_grid()

results = pd.DataFrame(
    columns=[
        "score",
        "time",
        "samples",
        "score_std",
        "subject",
        "session",
        "channels",
        "n_sessions",
        "dataset",
        "pipeline",
        "param_comb",  # List of model class kwargs used
    ]
)


#############################################################################
# Iterate over the parameter grid

for model_class_kwargs in param_grid:
    print(f"Processing model class kwargs: {model_class_kwargs}...")

    ##############################################################################
    # Create band-pass filters ensemble

    bpfe_mean = BandPassFilterEnsemble(frec_ranges=config.FREQ_BANDS_RANGES, sfreq=sfreq)
    bpfe_choquet = BandPassFilterEnsemble(frec_ranges=config.FREQ_BANDS_RANGES, sfreq=sfreq)
    bpfe_sugeno = BandPassFilterEnsemble(frec_ranges=config.FREQ_BANDS_RANGES, sfreq=sfreq)

    ##############################################################################
    # Create CSP ensemble
    cspe_mean = CSPEnsemble(n_components=config.CSP_COMPONENTS, n_frec_ranges=len(config.FREQ_BANDS_RANGES))
    cspe_choquet = CSPEnsemble(n_components=config.CSP_COMPONENTS, n_frec_ranges=len(config.FREQ_BANDS_RANGES))
    cspe_sugeno = CSPEnsemble(n_components=config.CSP_COMPONENTS, n_frec_ranges=len(config.FREQ_BANDS_RANGES))

    ##############################################################################
    # Create Model Ensembles Block

    # Create the model ensembles block
    clf_mean = IntvlMeanEnsemble.create_ensemble(
        model_class_list=config.MODEL_TYPES_LIST,
        model_class_names=config.MODEL_CLASS_NAMES,
        n_frec_ranges=len(config.FREQ_BANDS_RANGES),
        model_class_kwargs=model_class_kwargs,
        alpha=config.K_ALPHA,
        beta=config.K_BETA,
    )

    clf_choquet = IntvlChoquetEnsemble.create_ensemble(
        model_class_list=config.MODEL_TYPES_LIST,
        model_class_names=config.MODEL_CLASS_NAMES,
        n_frec_ranges=len(config.FREQ_BANDS_RANGES),
        model_class_kwargs=model_class_kwargs,
        alpha=config.K_ALPHA,
        beta=config.K_BETA,
    )

    clf_sugeno = IntvlSugenoEnsemble.create_ensemble(
        model_class_list=config.MODEL_TYPES_LIST,
        model_class_names=config.MODEL_CLASS_NAMES,
        n_frec_ranges=len(config.FREQ_BANDS_RANGES),
        model_class_kwargs=model_class_kwargs,
        alpha=config.K_ALPHA,
        beta=config.K_BETA,
    )

    ##############################################################################
    # Pipeline
    pipeline_mean = make_pipeline(bpfe_mean, cspe_mean, clf_mean)
    pipeline_choquet = make_pipeline(bpfe_choquet, cspe_choquet, clf_choquet)
    pipeline_sugeno = make_pipeline(bpfe_sugeno, cspe_sugeno, clf_sugeno)

    ##############################################################################
    # Choose paradigm and evaluation

    paradigm = LeftRightImagery()

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=f"{config.DISK_PATH}/models/" if config.SAVE_HDF5 else None,
        n_jobs=-1,
        n_jobs_evaluation=-1,
        random_state=config.SEED,
        additional_columns=["score_std"],
    )

    partial_results = evaluation.process(
        {
            "IntvlMeanEnsemble": pipeline_mean,
            "IntvlChoquetEnsemble": pipeline_choquet,
            "IntvlSugenoEnsemble": pipeline_sugeno,
        }
    )

    # Add parameter combination to the results
    partial_results["param_comb"] = [model_class_kwargs] * len(partial_results)
    partial_results["freq_bands_ranges"] = [config.FREQ_BANDS_RANGES] * len(partial_results)

    ##############################################################################
    # Save results to disk

    # Concatenate partial results to the final results dataframe
    results = pd.concat([results, partial_results], ignore_index=True)

    save_results_csv(results, f"{config.DISK_PATH}/{os.getenv('BNCI2014_001_GS_RESULTS')}", overwrite=True)

    print(f"Finished processing model class kwargs: {model_class_kwargs}.\n\n")
