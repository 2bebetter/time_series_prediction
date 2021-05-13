# -*- coding: utf-8 -*-
import config as cnf
import os
import pandas as pd
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures
import matplotlib.pylab as plt
import seaborn as sns
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(filename, time_index='Time', time_series='Inttraffic'):
    """
    load data from csv file
    :return:
    """
    df = pd.read_csv(filename, sep=',', header=0)
    ts = df[time_series]
    ts = ts.astype('float32')
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    extract features from time series selected for their relevance to forecasting
    dataframe assumed to be in tsfresh compatible format
    Params:
    -------
    df : dataframe from which to extract time series features
    Returns:
    -------
    features_filtered : numpy array containing the
    """
    if os.path.isfile('dataset/inttraffic_extracted_features.csv'):
        print('#### Features file exist - loading #######')
        extracted_features = pd.read_csv('dataset/inttraffic_extracted_features.csv')
        return extracted_features
    else:
        print('#### Features file does not exist - running feature extraction #######')
        # needs to be done for each time series
        extract_settings = ComprehensiveFCParameters()
        # 以id聚合
        extracted_features = extract_features(df, column_id='id', column_sort='Time', default_fc_parameters=extract_settings,
                             impute_function=impute)
        extracted_features.to_csv('dataset/inttraffic_extracted_features.csv')


df = load_data('dataset/inttraffic-features.csv')
print(type(df), df)

#特征提取
extract_settings = ComprehensiveFCParameters()
#以id聚合

X = generate_features(df)
print(X)

df_shift, y = make_forecasting_frame(df['Inttraffic'], kind='gmv', max_timeshift=24, rolling_direction=1)

print(y.shift(1))
print(y[1])
print(df_shift)
#提取最相关特征 三个步骤
X_filtered = extract_relevant_features(df_shift, y=y, column_id='id', column_sort='Time', default_fc_parameters=extract_settings)
X_filtered.info()
X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, test_size=4)

cl =DecisionTreeClassifier()
cl.fit(X_train, y_train)

print(classification_report(y_test, cl.predict(X_test)))

cl.n_features_

cl2 = DecisionTreeClassifier()
cl2.fit(X_filtered_train, y_train)
print(classification_report(y_test, cl2.predict(X_filtered_test)))


def generate_features_bak(df: pd.DataFrame) -> pd.DataFrame:
    """
    extract features from time series selected for their relevance to forecasting
    dataframe assumed to be in tsfresh compatible format
    Params:
    -------
    df : dataframe from which to extract time series features
    Returns:
    -------
    features_filtered : numpy array containing the
    """
    if os.path.isfile('inttraffic' + '{}_extracted_features.csv' \
            .format(cnf.RUN_TYPE)):
        print('#### Features file exist - loading #######')
        extracted_features = pd.read_csv(cnf.DATA + '{}_extracted_features.csv' \
                                         .format(cnf.RUN_TYPE))
        extracted_features.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
        extracted_features.set_index('Index', inplace=True)
        standard_scaler = preprocessing.StandardScaler()
        extracted_features_scaled = pd.DataFrame(standard_scaler.fit_transform(extract_features.values),
                                                 columns=extract_features.columns,
                                                 index=extract_features.index)

        return extracted_features_scaled
    else:
        print('#### Features file does not exist - running feature extraction #######')
        # needs to be done for each time series

        l = list(df['V1'].unique())  # getting all different time series from the list
        fc_param = dict()
        print('####  creating forecasting frame ####')
        for elm in l:
            print('#### Extr. and selecting features for\
					 series {} of {} ####' \
                  .format(l.index(elm) + 1, len(l)))
            df_tmp = df[df['V1'] == elm]
            df_fc, y = make_forecasting_frame(df_tmp['value'], kind=elm,
                                              rolling_direction=1,
                                              max_timeshift=7)

            extracted_features = extract_features(df_fc,
                                                  column_id='id',
                                                  column_sort='time',
                                                  column_value='value',
                                                  impute_function=impute,
                                                  default_fc_parameters=EfficientFCParameters())

            # verify matching index structure
            if y.index[0] in extracted_features.index:
                # do nothing as the indexes are in the same structure
                pass
            else:
                # modify y index to match extracted features index
                y.index = pd.MultiIndex.from_tuples(zip(['id'] * len(y.index), y.index))

            selected_features = select_features(extracted_features, y)

            fc_param_new = from_columns(selected_features)

            # Python 3.9 operation to unionize dictionaries
            fc_param = fc_param | fc_param_new
            fc_param_t = dict()
            # extracting
            for key in fc_param['value']:
                fc_param_t.update({key: fc_param['value'][key]})

        print('#### Extracting relevant fts for all series ####')

        extracted_features = extract_features(df,
                                              column_id='V1',
                                              column_sort='timestamp',
                                              column_value='value',
                                              impute_function=impute,
                                              default_fc_parameters=fc_param_t)

        standard_scaler = preprocessing.StandardScaler()
        extracted_features_scaled = pd.DataFrame(standard_scaler.fit_transform(extract_features.values),
                                                 columns=extract_features.columns,
                                                 index=extract_features.index)

        extracted_features.to_csv(cnf.DATA + '{}_extracted_features.csv'
                                  .format(cnf.RUN_TYPE))
        extracted_features_scaled.to_csv(cnf.Data + '{}_extr_features_scaled.csv'
                                         .scaled(cnf.RUN_TYPE))

    return extracted_features_scaled


def make_fc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    creates rolling window dataframe
    to be used inside apply of groupby
    """
    ts_id = df.iloc[0]['V1']
    df_res, y = make_forecasting_frame(df['value'],
                                       kind=ts_id,
                                       rolling_direction=1,
                                       max_timeshift=cnf.MAX_TIMESHIFT)
    df_res['y'] = y
    return df_res
