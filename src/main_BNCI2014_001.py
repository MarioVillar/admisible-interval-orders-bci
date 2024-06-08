import os
import warnings
import mne
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearnex import patch_sklearn

import config
from utils.disk import save_results_csv

from preprocessing.band_pass_filters import BandPassFilterEnsemble
from preprocessing.csp_ensemble import CSPEnsemble

from model.intvl_choquet_ensemble import IntvlChoquetEnsemble
from model.intvl_mean_ensemble import IntvlMeanEnsemble
from model.intvl_sugeno_ensemble import IntvlSugenoEnsemble

from evaluation.grid_param_search import get_best_params

#############################################################################
mne.set_log_level("warning")
moabb.set_log_level("ERROR")
warnings.filterwarnings("ignore")
patch_sklearn()


##############################################################################
# Initializing Datasets

dataset = BNCI2014_001()
# dataset.subject_list = [1]


##############################################################################
# Get the best parameters from the grid search
best_params = get_best_params(pd.read_csv(f"{config.DISK_PATH}/{os.getenv('BNCI2014_001_GS_RESULTS')}"))


##############################################################################
# Create band-pass filters ensemble

# Obtain sample frequency of the data
subject_data = dataset.get_data(subjects=[1])
sfreq = subject_data[1][list(subject_data[1].keys())[0]]["0"].info["sfreq"]

bpfe_mean = BandPassFilterEnsemble(
    frec_ranges=best_params["IntvlMeanEnsemble"]["freq_bands_ranges"], sfreq=sfreq, n_jobs=config.N_JOBS
)
bpfe_choquet = BandPassFilterEnsemble(
    frec_ranges=best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"], sfreq=sfreq, n_jobs=config.N_JOBS
)
bpfe_choquet_n_best = BandPassFilterEnsemble(
    frec_ranges=best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"], sfreq=sfreq, n_jobs=config.N_JOBS
)
bpfe_sugeno = BandPassFilterEnsemble(
    frec_ranges=best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"], sfreq=sfreq, n_jobs=config.N_JOBS
)


##############################################################################
# Create CSP ensemble
cspe_mean = CSPEnsemble(
    n_components=config.CSP_COMPONENTS,
    n_frec_ranges=len(best_params["IntvlMeanEnsemble"]["freq_bands_ranges"]),
    n_jobs=config.N_JOBS,
)
cspe_choquet = CSPEnsemble(
    n_components=config.CSP_COMPONENTS,
    n_frec_ranges=len(best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"]),
    n_jobs=config.N_JOBS,
)
cspe_choquet_n_best = CSPEnsemble(
    n_components=config.CSP_COMPONENTS,
    n_frec_ranges=len(best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"]),
    n_jobs=config.N_JOBS,
)
cspe_sugeno = CSPEnsemble(
    n_components=config.CSP_COMPONENTS,
    n_frec_ranges=len(best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"]),
    n_jobs=config.N_JOBS,
)


##############################################################################
# Create Model Ensembles Block


# Create the model ensembles block
clf_mean = IntvlMeanEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlMeanEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlMeanEnsemble"]["param_comb"],
    n_jobs=config.N_JOBS,
)

clf_choquet = IntvlChoquetEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlChoquetEnsemble"]["param_comb"],
    alpha=config.K_ALPHA,
    beta=config.K_BETA,
    n_jobs=config.N_JOBS,
)

clf_choquet_n_best = IntvlChoquetEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlChoquetEnsemble"]["param_comb"],
    alpha=config.K_ALPHA,
    beta=config.K_BETA,
    choquet_n_permu=config.N_ADMIS_PERMU,
    n_jobs=config.N_JOBS,
)

clf_sugeno = IntvlSugenoEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlSugenoEnsemble"]["param_comb"],
    n_jobs=config.N_JOBS,
)

##############################################################################
# Create simple model

lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf_ind = Pipeline([("CSP", csp), ("LDA", lda)])


##############################################################################
# Pipeline
pipeline_mean = make_pipeline(bpfe_mean, cspe_mean, clf_mean)
pipeline_choquet = make_pipeline(bpfe_choquet, cspe_choquet, clf_choquet)
pipeline_choquet_n_best = make_pipeline(bpfe_choquet_n_best, cspe_choquet_n_best, clf_choquet_n_best)
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
    additional_columns=["score_std"] + [f"scores_cv_{i}" for i in range(5)],
)

results = evaluation.process(
    {
        "IntvlMeanEnsemble": pipeline_mean,
        "IntvlChoquetEnsemble": pipeline_choquet,
        "IntvlChoquetEnsembleNBest": pipeline_choquet_n_best,
        "IntvlSugenoEnsemble": pipeline_sugeno,
        "CSP-LDA": clf_ind,
    }
)


param_comb_list = []
freq_bands_ranges_list = []
scores_cv_list = []

for i in range(len(results)):
    pipeline = results.loc[i, "pipeline"]

    if pipeline == "CSP-LDA":
        param_comb_list.append({})
        freq_bands_ranges_list.append([])
    else:
        if pipeline == "IntvlChoquetEnsembleNBest":
            pipeline = "IntvlChoquetEnsemble"
        param_comb_list.append(best_params[pipeline]["param_comb"])
        freq_bands_ranges_list.append(best_params[pipeline]["freq_bands_ranges"])

    # Undo the sloppy change made in MOABB to record all the cv scores
    scores_cv_list.append([results.iloc[i][f"scores_cv_{j}"] for j in range(5)])

results = results.drop([f"scores_cv_{i}" for i in range(5)], axis=1)

results["param_comb"] = param_comb_list
results["freq_bands_ranges"] = freq_bands_ranges_list
results["scores_cv"] = scores_cv_list


##############################################################################
# Save results to disk

if config.SAVE_TO_DISK:
    save_results_csv(results, f"{config.DISK_PATH}/{os.getenv('BNCI2014_001_MAIN_RESULTS')}", overwrite=False)
