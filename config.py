"""
This module contains the configuration settings
"""
#import pandas as pd

HOURLY_TRAIN = "dataset/Train/Hourly-train.csv"
HOURLY_TEST = "dataset/Test/Hourly-test.csv"

DAILY_TRAIN = "dataset/Train/Daily-train.csv"
DAILY_TEST = "dataset/Test/Daily-test.csv"

WEEKLY_TRAIN = "dataset/Train/Weekly-train.csv"
WEEKLY_TEST = "dataset/Test/Weekly-test.csv"

MONTHLY_TRAIN = "dataset/Train/Monthly-train.csv"
MONTHLY_TEST = "dataset/Test/Monthly-test.csv"

QUARTERLY_TRAIN = "dataset/Train/Quarterly-train.csv"
QUARTERLY_TEST = "dataset/Test/Quarterly-test.csv"

YEARLY_TRAIN = "dataset/Train/Yearly-train.csv"
YEARLY_TEST = "dataset/Test/Yearly-test.csv"

#
DATA = 'dataset/data/'

# run type should match test series being executed
RUN_TYPE = 'daily'
CUR_RUN_TRAIN = DAILY_TRAIN
CUR_RUN_TEST = DAILY_TEST

SAMPLING = True

SAMPLING_RATE = 0.01

LAGS = [1, 2, 3, 4, 5, 6, 7]

STEPS_AHEAD = 7

MAX_CLUSTER = 20

MODELS_PATH = 'models'

SCORES_FOLDER_PATH = 'scores'

SIL_SCORES_FILE_PATH = RUN_TYPE+'_silhouette_scores.txt'

FORECASTING_RES_FILE_NAME = RUN_TYPE+'_fc_results.csv'

INERTIA_FILE_PATH = RUN_TYPE+'_inertias.txt'

TOP_MODELS = 3

MAX_TIMESHIFT = 7

KFOLD = 10


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)