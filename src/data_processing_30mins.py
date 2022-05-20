#!/usr/bin/python3
# coding: utf-8

import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm as tqdm

DATA_PATH = Path("../Data/train_features")
LABEL_PATH = Path("../Data")

OUT_PATH = Path("../out")

start_timestamp = "2020-11-06T23:00:00"

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

def processData():
    directories = prepareDirectories()
    print("Opening Labels...")
    labels = pd.read_csv(LABEL_PATH/"open_train_labels.csv", parse_dates=["timestamp"])

    for code, path in directories.items():
        print(f'Processing data for {code}')
        processAirport(code, path, labels.copy())
    

def processAirport(code:str, path:Path, labels):
    data_files = list(path.glob("*"))

    weather_file = f"{code}_lamp.csv.bz2"
    weather_path = path / weather_file

    airport_weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    airport_weather = airport_weather.sort_values('timestamp').drop_duplicates()

    unique_forecast_time = airport_weather.timestamp.unique()
    print(f'Working with {len(unique_forecast_time)} unique forecasts for {code}')

    df_weather_forecast90 = pd.DataFrame()
    for i in tqdm(unique_forecast_time):
        df_col = airport_weather[airport_weather['timestamp']== i]
        df_col = df_col.sort_values('forecast_timestamp')
        df_col_forecast = df_col.iloc[1]
        df_weather_forecast90 = df_weather_forecast90.append(df_col_forecast)

    weather_forecast90_inuse = df_weather_forecast90.loc[ pd.to_datetime(df_weather_forecast90['forecast_timestamp']) >= pd.to_datetime(start_timestamp)]
    
    label_30min_katl = labels[(labels['airport']==code)&(labels['active']==1) & (labels['lookahead']==30)]

    weather_forecast90_inuse['timestamp'] = weather_forecast90_inuse['timestamp'] + datetime.timedelta(hours=0.5)

    forecast30_config_df = label_30min_katl.merge(weather_forecast90_inuse,on='timestamp',how='left')

    features_new, colNames_new = timeSeriesCreation(forecast30_config_df['timestamp'],precision_high=True)
    forecast30_config_df = pd.concat([forecast30_config_df,features_new],axis=1,join='outer')

    # forecast30_config_df_dropna = forecast30_config_df.dropna()
    forecast30_config_df_dropna = forecast30_config_df
    X_train = forecast30_config_df_dropna[['config','timestamp','year','month','day','hour','quarter','weekofyear',
                                           'dayofweek','weekend','hour_section','wind_direction','wind_speed',
                                           'temperature','wind_gust','cloud_ceiling','visibility']]

    # y_train = forecast30_config_df_dropna['config']

    X_train.to_csv(OUT_PATH/(code+"_x_train.csv"),index=False)
    # y_train.to_csv(OUT_PATH/(code+"_y_train.csv"),index=False)

def prepareDirectories():
    airport_directories = {}

    for path in DATA_PATH.glob("k*"):
        _id = path.name
        airport_directories[_id] = path
    print(f'Found {len(airport_directories)} directories')

    return airport_directories

processData()
