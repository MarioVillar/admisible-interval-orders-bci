import os
import dotenv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from utils.disk import get_disk_path


dotenv.load_dotenv()

# Random seed for reproducibility
SEED = 10


# Frequency band intervals in which the EEG signal is going to be decomposed
FREQ_BANDS_RANGES = [(7, 13), (11, 17), (15, 21), (19, 25), (23, 30)]


FREQ_BANDS = [6, 10, 8, 15, 14, 28, 24, 35]

# Number of components to be extracted from the CSP algorithm
CSP_COMPONENTS = 4

# Value of alpha and beta in the K-alpha and K-beta mappings for ordering of intervals
K_ALPHA = 0.5
K_BETA = 0.6

# Model type list to be replicated in the model ensembles
MODEL_TYPES_LIST = [SVC, RandomForestClassifier, LinearDiscriminantAnalysis]

# Model class names
MODEL_CLASS_NAMES = ["svc", "rfc", "lda"]

# Model class kwargs
MODEL_CLASS_KWARGS = [{"C": 0.1, "kernel": "linear", "probability": True}, {}, {}]


# Cross-validation parameters
CV_PARAMS_SVC = {"C": [0.01, 0.1, 1, 10], "kernel": ["linear", "rbf"], "probability": [True]}

CV_PARAMS_RFC = {"n_estimators": [50, 100, 200, 300], "max_depth": [None, 10, 20], "random_state": [SEED]}


# Whether to save the results to disk or not
SAVE_TO_DISK = os.getenv("SAVE_TO_DISK").lower() == "true"

# Whether to save the models to disk using HDF5 or not
SAVE_HDF5 = os.getenv("SAVE_HDF5").lower() == "true"

# Path to save the results
DISK_PATH = get_disk_path()
