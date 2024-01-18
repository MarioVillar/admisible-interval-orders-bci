from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Frequency band intervals in which the EEG signal is going to be decomposed
FREQ_BANDS_RANGES = [(6, 10), (8, 15), (14, 28), (24, 35)]
FREQ_BANDS = [6, 10, 8, 15, 14, 28, 24, 35]

# Number of components to be extracted from the CSP algorithm
CSP_COMPONENTS = 4

# Value of alpha in the K-alpha mappings of intervals
K_ALPHA = 0.5

# Model type list to be replicated in the model ensembles
MODEL_TYPES_LIST = [SVC, RandomForestClassifier]

# Model class names
MODEL_CLASS_NAMES = ["svc", "rfc"]

# Model class kwargs
MODEL_CLASS_KWARGS = [{"C": 0.1, "kernel": "linear", "probability": True}, {}]


# Whether to save the results to disk or not
SAVE_TO_DISK = True
