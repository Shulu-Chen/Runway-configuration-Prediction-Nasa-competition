#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 4/22/2022 6:56 PM
 @Author  : Chelsea
 @FileName: generate_submission_x.py
 @Software: PyCharm
'''

import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np

from pandas import to_datetime, read_csv

DATA_PATH = Path("../data/train_features")
LABEL_PATH = Path("../data")

OUT_PATH = Path("out")

MAXLOOKAHEAD = 360 # mins
skip = []


# select airport code and path into a list
def prepareDirectories():
    airport_directories = {}

    for path in DATA_PATH.glob("k*"):
        _id = path.name
        airport_directories[_id] = path
    print(f'Found {len(airport_directories)} directories')

    return airport_directories

# generate time-related features
def deconstructDate(date):
    _month = date.strftime("%m")
    _day = date.strftime("%d")
    _hour = date.strftime("%H")
    _min = date.strftime("%M")

    return(_month,_day,_hour,_min)


def processData():
    directories = prepareDirectories()
    print("Opening Labels...")
    labels = read_csv(LABEL_PATH/"open_submission_format.csv", parse_dates=["timestamp"])

    for code, path in directories.items():
        if code in skip:
            continue
        print(f'Processing data for {code}')
        processAirport(code, path, labels[(labels["airport"] == code)].copy())


# generate weather features and time related features of each airport in training dataset
def processAirport(code:str, path:Path, labels):

    min_label_time, max_label_time = min(to_datetime(labels["timestamp"])),max(to_datetime(labels["timestamp"]))
    data_files = list(path.glob("*"))

    weather_file = f"{code}_lamp.csv.bz2"
    weather_path = path / weather_file

    airport_weather = read_csv(weather_path, parse_dates=["timestamp"])
    airport_weather = airport_weather[(to_datetime(airport_weather["timestamp"]) >= min_label_time-pd.DateOffset(hours=1))&(to_datetime(airport_weather["timestamp"]) <= max_label_time)]

    # check and impute missing value
    airport_weather.info()
    airport_weather[airport_weather['temperature'].isna()]['timestamp'].unique()

    lamp_nan_index = airport_weather[airport_weather['temperature'].isna()].index
    for i in lamp_nan_index:
        forecast_timestamp = airport_weather.loc[i]['forecast_timestamp']
        airport_weather.loc[i]= airport_weather[airport_weather['forecast_timestamp']==forecast_timestamp].iloc[-2]


    lamp_nan_index = airport_weather[airport_weather['temperature'].isna()].index
    for i in lamp_nan_index:
        forecast_timestamp = airport_weather.loc[i]['forecast_timestamp']
        airport_weather.loc[i]= airport_weather[airport_weather['forecast_timestamp']==forecast_timestamp].iloc[-3]


    lamp_nan_index = airport_weather[airport_weather['temperature'].isna()].index
    for i in lamp_nan_index:
        forecast_timestamp = airport_weather.loc[i]['forecast_timestamp']
        airport_weather.loc[i]= airport_weather[airport_weather['forecast_timestamp']==forecast_timestamp].iloc[-4]

    airport_weather[airport_weather['temperature'].isna()]


    # label_timestamp, lookahead, forecast_timestamp, forecast_diff, [forecast details]

    label_timestamps = labels.timestamp.unique()

    dataset = []

    for l_timestamp in tqdm(label_timestamps):
        valid_forecasts = None
        hour_offset = 1

        l_time = to_datetime(l_timestamp)


        temp_valid = airport_weather[(to_datetime(airport_weather["timestamp"]) <= l_time)&(to_datetime(airport_weather["timestamp"]) >= l_time-pd.DateOffset(hours=8))]
        if len(temp_valid) > 0:
            valid_forecasts = temp_valid

        if valid_forecasts is None:
            continue

        for lookahead in range(30, MAXLOOKAHEAD+1, 30):
            v_forecast_time = valid_forecasts["forecast_timestamp"].to_numpy()

            idx = (np.abs(to_datetime(v_forecast_time)-(l_time+pd.DateOffset(minutes=lookahead)))).argmin()
            v = valid_forecasts.to_numpy()[idx]

            lookahead_time = l_time+pd.DateOffset(minutes=lookahead)

            lookahead_month, lookahead_day, lookahead_hour, lookahead_min = deconstructDate(lookahead_time)

            to_add = [str(l_time), lookahead, lookahead_month, lookahead_day, lookahead_hour, lookahead_min]
            to_add.extend(v[i] for i in range(2, len(v)))
            to_add.insert(6, ((l_time+pd.DateOffset(minutes=lookahead))-to_datetime(v[1])).delta/6e10)
            label = labels[(to_datetime(labels['timestamp']) == l_timestamp)&(labels['lookahead']==lookahead)].to_numpy()[0,3]
            to_add.append(label)

            dataset.append(np.array(to_add))

    data_frame = pd.DataFrame(dataset,columns=['label_timestamp', 'lookahead', 'l_month', 'l_day', 'l_hour', 'l_min', 'forecast_diff', 'temperature', 'wind_direction','wind_speed','wind_gust','cloud_ceiling','visibility','cloud','lightning_prob','precip','config'])
    X_train = data_frame[['label_timestamp', 'lookahead', 'l_month', 'l_day', 'l_hour', 'l_min', 'forecast_diff','wind_direction','wind_speed','temperature','wind_gust','cloud_ceiling','visibility']]
    y_train = data_frame[['label_timestamp','lookahead','config']]

    X_train.to_csv(f'../out/{code}_timebased_x_test.csv', index=False)
    y_train.to_csv(f'../out/{code}_timebased_y_test.csv', index=False)


processData()
