# This example is a reproduction of the example in the following link:
#   https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
#
# The idea of this example is to analyze the variations induced in the acccuracy of the model
#   due to varying time intervals.
# The code has been taken fromt he aforementioned link, changing the used pipeline
#   to the one proposed in this project.


import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline

from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf


import config
from preprocessing.band_pass_filters import BandPassFilterEnsemble
from preprocessing.csp_ensemble import CSPEnsemble
from model.intvl_choquet_ensemble import IntvlChoquetEnsemble
from model.intvl_mean_ensemble import IntvlMeanEnsemble

print(__doc__)

# #############################################################################
# # Set parameters and read data

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
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

##################################################################################
# This part differs from the original example because the pipeline used
#   is the one proposed in this project.

sfreq = raw.info["sfreq"]

# Create band-pass filters ensemble
bpfe = BandPassFilterEnsemble(frec_ranges=config.FREQ_BANDS_RANGES, sfreq=sfreq)

# Create CSP ensemble
cspe = CSPEnsemble(n_components=config.CSP_COMPONENTS, n_frec_ranges=len(config.FREQ_BANDS_RANGES))

# Create the model ensemble
mdes = IntvlChoquetEnsemble.create_ensemble(
    model_class_list=config.MODEL_TYPES_LIST,
    model_class_names=config.MODEL_CLASS_NAMES,
    n_frec_ranges=len(config.FREQ_BANDS_RANGES),
    model_class_kwargs=config.MODEL_CLASS_KWARGS,
    alpha=config.K_ALPHA,
)

# mdes = IntvlMeanEnsemble.create_ensemble(
#     model_class_list=config.MODEL_TYPES_LIST,
#     model_class_names=config.MODEL_CLASS_NAMES,
#     n_frec_ranges=len(config.FREQ_BANDS_RANGES),
#     model_class_kwargs=config.MODEL_CLASS_KWARGS,
# )

clf = make_pipeline(bpfe, cspe, mdes)

##################################################################################

scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=-1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))


w_length = int(sfreq * 0.5)  # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    # fit classifier
    clf.fit(epochs_data_train[train_idx], y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        score_this_window.append(clf.score(epochs_data[test_idx][:, :, n : (n + w_length)], y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
plt.axvline(0, linestyle="--", color="k", label="Onset")
plt.axhline(0.5, linestyle="-", color="k", label="Chance")
plt.xlabel("time (s)")
plt.ylabel("classification accuracy")
plt.title("Classification score over time")
plt.legend(loc="lower right")
plt.show()
