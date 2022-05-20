#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 3/28/2022 22:15 PM
 @Author  : Chelsea
 @FileName: data_process.py
 @Software: PyCharm
'''
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import datetime
from sklearn.preprocessing import OneHotEncoder

# ===================================================== read in airport ==================================================
warnings.filterwarnings('ignore')


DATA_PATH = Path("data/train_features")
start_time_stamp = "2020-11-06T23:00:00"
airport = "katl"

airport_directories = sorted(path for path in DATA_PATH.glob("k*"))

airport_directory = airport_directories[0]
data_files = list(airport_directory.glob("*"))

airport_directory = airport_directories[0]

# ======================================================== Lamp weather data ==================================================
airport_code = airport_directory.name
filename = f"{airport_code}_lamp.csv.bz2"
filepath = airport_directory / filename

airport_lamp = pd.read_csv(filepath, parse_dates=["timestamp"])

airport_lamp=airport_lamp.sort_values('timestamp')

airport_lamp=airport_lamp.drop_duplicates()

unique_forecast_time = airport_lamp.timestamp.unique()


# weather data in use

df_weather_forecast90 = pd.DataFrame()
for i in unique_forecast_time:
    df_col = airport_lamp[airport_lamp['timestamp']== i]
    df_col = df_col.sort_values('forecast_timestamp')
    df_col_forecast = df_col.iloc[1]
    df_weather_forecast90 = df_weather_forecast90.append(df_col_forecast)

# weather_forecast90_inuse = df_weather_forecast90.loc[3574:225646]
weather_forecast90_inuse = df_weather_forecast90.loc[ pd.to_datetime(df_weather_forecast90['forecast_timestamp']) >= pd.to_datetime(start_time_stamp)]


# ================================================================== training data - runway config ===========================================================
DATA_PATH = Path("data/open_train_labels")
open_train_labels = pd.read_csv(DATA_PATH / "open_train_labels.csv.bz2", parse_dates=["timestamp"])

label_30min_katl = open_train_labels[(open_train_labels['active']==1) & (open_train_labels['lookahead']==30)&(open_train_labels['airport']==airport)]
train_label_unique = np.unique(label_30min_katl['config'])



# ================================================================== Actual runway config ====================================================

airport_code = airport_directory.name
filename = f"{airport_code}_airport_config.csv.bz2"
filepath = airport_directory / filename

airport_config = pd.read_csv(filepath, parse_dates=["timestamp"])

# Enlabel actual runway config to train label

for i in range(0,len(train_label_unique)):
    train_label_unique[i]=train_label_unique[i].replace('katl:','')  # train_label_unique is open training label

# select 'other' config
other = np.setdiff1d(airport_config['airport_config'],train_label_unique)
for idx,x in enumerate(airport_config['airport_config']):
    if x in other:
        airport_config['airport_config'].iloc[idx] = 'other'

# np.unique(airport_config['airport_config'])


# ======================================================================== time-related function  ==============================================================
def timeSeriesCreation(timeSeries, timeStamp=None, precision_high=False):

    features_new = pd.DataFrame()

    timeSeries = pd.to_datetime(timeSeries)

    features_new['year'] = timeSeries.dt.year
    features_new['month'] = timeSeries.dt.month
    features_new['day'] = timeSeries.dt.day

    if precision_high != False:
        features_new['hour'] = timeSeries.dt.hour
        features_new['minute'] = timeSeries.dt.minute
        features_new['second'] = timeSeries.dt.second

    features_new['quarter'] = timeSeries.dt.quarter
    features_new['weekofyear'] = timeSeries.dt.weekofyear
    features_new['dayofweek'] = timeSeries.dt.dayofweek + 1
    features_new['weekend'] = (features_new['dayofweek'] > 5).astype(int)

    if precision_high != False:
        features_new['hour_section'] = (features_new['hour'] // 6).astype(int)

    colNames_new = list(features_new.columns)
    return features_new, colNames_new


# ================================================================ feature prepare ====================================================================

weather_forecast90_inuse['timestamp'] = weather_forecast90_inuse['timestamp'] + datetime.timedelta(hours=0.5)
forecast30_config_df = label_30min_katl.merge(weather_forecast90_inuse,on='timestamp',how='left')


# select time features
features_new, colNames_new = timeSeriesCreation(forecast30_config_df['timestamp'],precision_high=True)
seq = ['month','day','hour','minute','second','dayofweek']
features_new1 = features_new[seq]

# # encode time charater
# seq_encode = ['month', 'day','hour']
# feature_seq_new = features_new1[seq_encode]
#
# enc = OneHotEncoder()
# features_seq_new=enc.fit_transform(feature_seq_new).toarray()
# features_seq_new
#
# features_seq_new = pd.DataFrame(enc.fit_transform(feature_seq_new).toarray(),
#                                columns = cate_colName(enc, seq_new, drop=None))


forecast30_config_df = pd.concat([forecast30_config_df,features_new],axis=1,join='outer')



## MAR 28 ##
##================ HANDLE NA IN WEATHER DATA / ENCODE TIMESTAMP / ADD FEATURES(ACTUAL RUNWAY CONFIG) / WIND_DIRECTION DIFF ==========================


#  ACTUAL RUNWAY CONFIG for 30 mins
act_config = []
for i in range(len(label_30min_katl)):
    act_config.append(airport_config[label_30min_katl['timestamp'].iloc[i] > airport_config['timestamp']]
                                   .iloc[-1]['airport_config'])
    act_config=pd.DataFrame(act_config)
    act_config.columns = ['act_config']

forecast30_config_df = pd.concat([forecast30_config_df,act_config],axis=1,join='outer')

forecast30_config_df_dropna = forecast30_config_df.dropna()

# # HANDLE NA IN WEATHER DATA
# forecast30_config_df.isnull().sum()
# wind_dirc_nan = forecast30_config_df[forecast30_config_df['wind_direction'].isna()].head(30)
# wind_dirc_nan
# wind_dirc_nan.index
# for idx, wind in enumerate(wind_dirc_nan['wind_direction']):

# ================================================================================ Training models =========================================================


# X_train_1 = forecast30_config_df_dropna[['wind_direction','wind_speed']]
#
# X_train = forecast30_config_df_dropna[['wind_direction','wind_speed','temperature','wind_gust','cloud_ceiling','visibility']]
# y_train = forecast30_config_df_dropna['config']
#
# X_train.to_csv(airport+"x_train.csv",index=False)
# y_train.to_csv(airport+"y_train.csv",index=False)


x_train_katl = pd.read_csv('out/katl_timebased_x_train.csv')
y_train_katl = pd.read_csv('out/katl_timebased_y_train.csv')

x_train_katl['label_timestamp'].type()