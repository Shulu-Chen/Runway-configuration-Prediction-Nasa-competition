#!/usr/bin/python3
# coding: utf-8

import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np

DATA_PATH = Path("data/train_features")
LABEL_PATH = Path("data/train_labels")

OUT_PATH = Path("out")

MAXLOOKAHEAD = 360 # mins
skip = []

def prepareDirectories():
    airport_directories = {}

    for path in DATA_PATH.glob("k*"):
        _id = path.name
        airport_directories[_id] = path
    print(f'Found {len(airport_directories)} directories')

    return airport_directories


def processData():
    directories = prepareDirectories()
    print("Opening Labels...")
    labels = pd.read_csv(LABEL_PATH/"open_train_labels.csv", parse_dates=["timestamp"])

    for code, path in directories.items():
        if code in skip:
            continue
        print(f'Processing data for {code}')
        processAirport(code, path, labels[(labels["airport"] == code)&(labels["active"] == 1)].copy())
    

def processAirport(code:str, path:Path, labels):

    min_label_time, max_label_time = min(pd.to_datetime(labels["timestamp"])),max(pd.to_datetime(labels["timestamp"]))
    data_files = list(path.glob("*"))

    weather_file = f"{code}_lamp.csv.bz2"
    weather_path = path / weather_file

    airport_weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    airport_weather = airport_weather[(pd.to_datetime(airport_weather["timestamp"]) >= min_label_time-pd.DateOffset(hours=1))&(pd.to_datetime(airport_weather["timestamp"]) <= max_label_time)]


    # label_timestamp, lookahead, forecast_timestamp, forecast_diff, [forecast details]

    label_timestamps = labels.timestamp.unique()

    dataset = []

    for l_timestamp in tqdm(label_timestamps):
        valid_forecasts = None
        hour_offset = 1

        l_time = pd.to_datetime(l_timestamp)

        while valid_forecasts is None and hour_offset <13:
            temp_valid = airport_weather[(pd.to_datetime(airport_weather["timestamp"]) <= l_time)&(pd.to_datetime(airport_weather["timestamp"]) >= l_time-pd.DateOffset(hours=hour_offset))]
            if len(temp_valid) > 0:
                valid_forecasts = temp_valid
            hour_offset += 1

        if valid_forecasts is None:
            continue

        for lookahead in range(30, MAXLOOKAHEAD+1, 30):
            v_forecast_time = valid_forecasts["forecast_timestamp"].to_numpy()

            idx = (np.abs(pd.to_datetime(v_forecast_time)-(l_time+pd.DateOffset(minutes=lookahead)))).argmin()
            v = valid_forecasts.to_numpy()[idx]

            to_add = [str(l_time), lookahead]
            to_add.extend(v[i] for i in range(1, len(v)))
            to_add.insert(3, ((l_time+pd.DateOffset(minutes=lookahead))-pd.to_datetime(to_add[2])).delta/6e10)
            label = labels[(pd.to_datetime(labels['timestamp']) == l_timestamp)&(labels['lookahead']==lookahead)].to_numpy()[0,3]
            to_add.append(label)

            dataset.append(np.array(to_add))

    data_frame = pd.DataFrame(dataset,columns=['label_timestamp', 'lookahead', 'forecast_timestamp', 'forecast_diff', 'temperature', 'wind_direction','wind_speed','wind_gust','cloud_ceiling','visibility','cloud','lightning_prob','precip','config'])
    X_train = data_frame[['label_timestamp', 'lookahead', 'forecast_timestamp', 'forecast_diff','wind_direction','wind_speed','temperature','wind_gust','cloud_ceiling','visibility']]
    y_train = data_frame[['label_timestamp','lookahead','config']]

    X_train.to_csv(OUT_PATH / f'{code}_timebased_x_train.csv', index=False)
    y_train.to_csv(OUT_PATH / f'{code}_timebased_y_train.csv', index=False)   





processData()
