# This example is a reproduction of the example in the following link:
#   https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
#
# The idea of this example is to analyze the variations induced in the acccuracy of the model
#   due to varying time intervals.
# The code has been taken fromt he aforementioned link, changing the used pipeline
#   to the one proposed in this project.


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from moabb import set_log_level as moabb_set_log_level
import warnings
import mne
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


import config
from preprocessing.band_pass_filters import BandPassFilterEnsemble
from preprocessing.csp_ensemble import CSPEnsemble
from model.intvl_choquet_ensemble import IntvlChoquetEnsemble
from model.intvl_mean_ensemble import IntvlMeanEnsemble
from model.intvl_sugeno_ensemble import IntvlSugenoEnsemble
from utils.disk import save_results_csv
from evaluation.grid_param_search import get_best_params


#############################################################################
mne.set_log_level("CRITICAL")
moabb_set_log_level("ERROR")
warnings.filterwarnings("ignore")


#############################################################################
# Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.0, 4.0
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)

# Apply band-pass filter
#   This step is avoided because it is implemented in the actual pipeline of this project
# raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
labels = epochs.events[:, -1] - 2


# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)


##################################################################################
# This part differs from the original example because the pipeline used
#   is the one proposed in this project.


##############################################################################
# Get the best parameters from the grid search
best_params = get_best_params(pd.read_csv(f"{config.DISK_PATH}/{os.getenv('BNCI2014_001_GS_RESULTS')}"))


##############################################################################
# Create band-pass filters ensemble

# Obtain sample frequency of the data
sfreq = raw.info["sfreq"]  # Obtain sample frequency of the data

bpfe_mean = BandPassFilterEnsemble(frec_ranges=best_params["IntvlMeanEnsemble"]["freq_bands_ranges"], sfreq=sfreq)
bpfe_choquet = BandPassFilterEnsemble(frec_ranges=best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"], sfreq=sfreq)
bpfe_sugeno = BandPassFilterEnsemble(frec_ranges=best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"], sfreq=sfreq)


##############################################################################
# Create CSP ensemble
cspe_mean = CSPEnsemble(
    n_components=config.CSP_COMPONENTS, n_frec_ranges=len(best_params["IntvlMeanEnsemble"]["freq_bands_ranges"])
)
cspe_choquet = CSPEnsemble(
    n_components=config.CSP_COMPONENTS, n_frec_ranges=len(best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"])
)
cspe_sugeno = CSPEnsemble(
    n_components=config.CSP_COMPONENTS, n_frec_ranges=len(best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"])
)


##############################################################################
# Create Model Ensembles Block


# Create the model ensembles block
clf_mean = IntvlMeanEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlMeanEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlMeanEnsemble"]["param_comb"],
    alpha=config.K_ALPHA,
)

clf_choquet = IntvlChoquetEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlChoquetEnsemble"]["param_comb"],
    alpha=config.K_ALPHA,
)

clf_sugeno = IntvlSugenoEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"]),
    model_class_kwargs=best_params["IntvlSugenoEnsemble"]["param_comb"],
    alpha=config.K_ALPHA,
)


##############################################################################
# Pipeline
clf_mean = make_pipeline(bpfe_mean, cspe_mean, clf_mean)
clf_choquet = make_pipeline(bpfe_choquet, cspe_choquet, clf_choquet)
clf_sugeno = make_pipeline(bpfe_sugeno, cspe_sugeno, clf_sugeno)


##################################################################################


w_length = int(sfreq * 0.5)  # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)


def score_clf(clf):
    scores_windows = []

    cv = ShuffleSplit(5, test_size=0.2, random_state=config.SEED)
    cv_split = cv.split(epochs_data_train)

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        # fit classifier
        clf.fit(epochs_data_train[train_idx], y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            score_this_window.append(clf.score(epochs_data[test_idx][:, :, n : (n + w_length)], y_test))
        scores_windows.append(score_this_window)

    return scores_windows


scores_mean = score_clf(clf_mean)
scores_choquet = score_clf(clf_choquet)
scores_sugeno = score_clf(clf_sugeno)

# Plot scores over time
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin


# Create df to save it to disk
results = pd.DataFrame(
    {
        "pipeline": ["IntvlMeanEnsemble", "IntvlChoquetEnsemble", "IntvlSugenoEnsemble"],
        "scores": [scores_mean, scores_choquet, scores_sugeno],
        "w_times": [w_times, w_times, w_times],
        "param_comb": [
            best_params["IntvlMeanEnsemble"]["param_comb"],
            best_params["IntvlChoquetEnsemble"]["param_comb"],
            best_params["IntvlSugenoEnsemble"]["param_comb"],
        ],
        "freq_bands_ranges": [
            best_params["IntvlMeanEnsemble"]["freq_bands_ranges"],
            best_params["IntvlChoquetEnsemble"]["freq_bands_ranges"],
            best_params["IntvlSugenoEnsemble"]["freq_bands_ranges"],
        ],
    }
)

##############################################################################
# Save results to disk
if config.SAVE_TO_DISK:
    save_results_csv(results, f"{config.DISK_PATH}/{os.getenv('EEGBCI_TIME_INTVLS_MAIN_RESULTS')}", overwrite=True)
