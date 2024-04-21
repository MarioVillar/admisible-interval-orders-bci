import pandas as pd
import os

import config

from visualization.plot_results import bar_plot_by_subject, line_plot_by_time_intvl


#############################################################################
# Read results csv files to DataFrames
BNCI2014_001_results = pd.read_csv(f"{config.DISK_PATH}/{os.getenv('BNCI2014_001_MAIN_RESULTS')}")
# acc_time_intvls_results = pd.read_csv(f"{config.DISK_PATH}/{os.getenv('EEGBCI_TIME_INTVLS_MAIN_RESULTS')}")


#############################################################################
# Plot BNCI2014_001 dataset results
barplot = bar_plot_by_subject(
    results=BNCI2014_001_results,
    plot_csp_lda=False,
    save_to_disk=config.SAVE_TO_DISK,
    img_name="BNCI2014_001_5_fold_roc_auc",
)


#############################################################################
# Plot EEGBCI dataset results
# lineplot = line_plot_by_time_intvl(
#     results=acc_time_intvls_results, save_to_disk=config.SAVE_TO_DISK, img_name="EEGBCI_time_intvls_5_fold_acc"
# )
