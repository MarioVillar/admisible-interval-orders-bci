import warnings

import matplotlib.pyplot as plt
import seaborn as sns

import mne

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import moabb
from moabb.datasets import BNCI2014_004, Zhou2016
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery

import constants

from preprocessing.test_sklearn_transformer import PrintTransformer
from preprocessing.band_pass_filters import BandPassFilterEnsemble
from preprocessing.csp_ensemble import CSPEnsemble
from preprocessing.time_frequency_filter import time_freq_filter_init

from model.intvl_choquet_ensemble import IntvlChoquetEnsemble


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")


##############################################################################
# Initializing Datasets

dataset = BNCI2014_004()
# dataset.subject_list = [1]


##############################################################################
# Create time frequency filter

# Obtain sample frequency of the data
# subject_data = dataset.get_data(subjects=[1])
# sfreq = subject_data[1][list(subject_data[1].keys())[0]]["0"].info["sfreq"]

# # Initialize the TimeFrequency object
# tft = time_freq_filter_init(sfreq=sfreq)


##############################################################################
# Create band-pass filters ensemble

# Obtain sample frequency of the data
subject_data = dataset.get_data(subjects=[1])
sfreq = subject_data[1][list(subject_data[1].keys())[0]]["0"].info["sfreq"]

bpfe = BandPassFilterEnsemble(frec_ranges=constants.FREQ_BANDS_RANGES, sfreq=sfreq)


##############################################################################
# Create CSP ensemble
cspe = CSPEnsemble(n_components=constants.CSP_COMPONENTS, n_frec_ranges=len(constants.FREQ_BANDS_RANGES))


##############################################################################
# Create Model Ensemble
model_types_list = [SVC, RandomForestClassifier]
model_class_kwargs = [{"C": 0.1, "kernel": "linear", "probability": True}, {}]
model_class_names = ["svc", "rfc"]

# Create the ensemble
icens = IntvlChoquetEnsemble.create_ensemble(
    model_class_list=model_types_list,
    model_class_names=model_class_names,
    n_frec_ranges=len(constants.FREQ_BANDS_RANGES),
    model_class_kwargs=model_class_kwargs,
)


##############################################################################
# Pipeline
pipeline = make_pipeline(bpfe, cspe, icens)


##############################################################################
# Choose paradigm and evaluation

paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset], overwrite=False, hdf5_path=None)

results = evaluation.process({"Model ensembles + Choquet + K-lambda": pipeline})


##############################################################################
# Plotting Results
# ----------------
# results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]

# g = sns.catplot(
#     kind="bar",
#     x="score",
#     y="subj",
#     hue="pipeline",
#     col="dataset",
#     height=12,
#     aspect=0.5,
#     data=results,
#     orient="h",
#     palette="viridis",
# )
# plt.show()
