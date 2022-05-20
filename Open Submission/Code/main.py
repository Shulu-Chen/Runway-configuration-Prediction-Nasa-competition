import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
from pandas import to_datetime, read_csv
import time
import warnings
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import sys
warnings.filterwarnings('ignore')


DATA_PATH = Path("../data/train_features")
LABEL_PATH = Path("../data")

OUT_PATH = Path("out")

MAXLOOKAHEAD = 360 # mins
skip = []
input_time_str = sys.argv[1]
input_time = datetime.datetime.strptime(input_time_str, '%Y-%m-%dT%H:%M:%S')

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


# generate weather features and time related features of each airport
def processAirport(code:str, path:Path):

    min_label_time, max_label_time = input_time,input_time
    data_files = list(path.glob("*"))

    weather_file = f"{code}_lamp.csv.bz2"
    weather_path = path / weather_file

    airport_weather = read_csv(weather_path, parse_dates=["timestamp"])

    valid_forecasts = airport_weather[(to_datetime(airport_weather["timestamp"]) >= min_label_time-pd.DateOffset(hours=1))&(to_datetime(airport_weather["timestamp"]) <= max_label_time)]
    dataset = []
    for lookahead in range(30, MAXLOOKAHEAD+1, 30):
        v_forecast_time = valid_forecasts["forecast_timestamp"].to_numpy()

        idx = (np.abs(to_datetime(v_forecast_time)-(input_time+pd.DateOffset(minutes=lookahead)))).argmin()
        v = valid_forecasts.to_numpy()[idx]
        lookahead_time = input_time+pd.DateOffset(minutes=lookahead)

        lookahead_month, lookahead_day, lookahead_hour, lookahead_min = deconstructDate(lookahead_time)

        to_add = [str(input_time), lookahead, lookahead_month, lookahead_day, lookahead_hour, lookahead_min]
        to_add.extend(v[i] for i in range(2, len(v)))
        to_add.insert(6, ((input_time+pd.DateOffset(minutes=lookahead))-to_datetime(v[1])).delta/6e10)

        dataset.append(np.array(to_add))

    data_frame = pd.DataFrame(dataset,columns=['label_timestamp', 'lookahead', 'l_month', 'l_day', 'l_hour', 'l_min', 'forecast_diff', 'temperature', 'wind_direction','wind_speed','wind_gust','cloud_ceiling','visibility','cloud','lightning_prob','precip'])
    X_train = data_frame[['label_timestamp', 'lookahead', 'l_month', 'l_day', 'l_hour', 'l_min', 'forecast_diff','wind_direction','wind_speed','temperature','wind_gust','cloud_ceiling','visibility']]
    # X_train.to_csv(f"{code}_test.csv")
    out = RunModel(X_train,code)
    return out

# load in y-test, model and run
def RunModel(X_train,airport):

    y_location = f'../Data/open_train_labels.csv.bz2'

    airport_configs = pd.read_csv(f'..\\Data\\train_features\\{airport}\\{airport}_airport_config.csv.bz2')

    X, y = X_train, pd.read_csv(y_location)
    unique_labels = y[y['airport'] == airport]['config'].unique()

    le = LabelEncoder()
    le.fit(unique_labels)
    unique_labels=sorted(unique_labels)

    #Reload models
    model_dir = f"../Models/{airport}_model.pkl"

    model = joblib.load(model_dir)
    print(f"Reload {airport} model successfully")

    out = run_test(X.to_numpy(), airport_configs, model, le, airport,unique_labels)
    return out


def run_test(X, airport_configs, model, encoder, airport,labels):
    pre_config = 0
    true, pred, pred_prob = [], [], []
    N = len(X)
    output = pd.read_csv(f"../Data/output.csv")
    # print(output)
    airport_list = [airport]*len(labels)

    for i in tqdm(range(N)):
        X_val = X[i]

        if X_val[1] == '30':
            time = pd.to_datetime(X_val[0])
            # print(time)
            pre_config = get_real_config(airport_configs, encoder, airport, time)
        # print("pre_config",pre_config)
        time_list = [pd.to_datetime(X_val[0])]*len(labels)
        forecast_list = [X_val[1]]*len(labels)
        X_val = np.array([np.append(X_val, pre_config)[1:]])
        # print(X_val)
        est_prob = make_prediction(X_val, model)[0]
        pre_config=np.argmax(est_prob)
        merge = pd.DataFrame(data = [airport_list, time_list,forecast_list,labels,est_prob],
                             index=['airport','timestamp','lookahead','config','active']).T
        output = pd.concat([output,merge])

    output = output.iloc[1:]
    # output['timestamp'] = output['timestamp'].str.replace(' ', 'T')
    output['timestamp'] = output['timestamp'].replace(' ', 'T')
    output.to_csv(f"../out/{airport}_stage2.csv")
    return output

# get actual runway config before 30 mins lookahead and encode them as a feature
def get_real_config(airport_configs, encoder, airport, time):

    valid = airport_configs[(pd.to_datetime(airport_configs["timestamp"]) <= time)].to_numpy()[-1,1]
    valid = f'{airport}:{valid}'
    # print(valid)
    try:
        encoded = encoder.transform([valid])
    except:
        encoded = encoder.transform([f'{airport}:other'])[0]

    return encoded

def make_prediction(X, model):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)

# run model and make prediction for each airport
def processData():
    directories = prepareDirectories()

    for code, path in directories.items():
        if code in skip:
            continue
        print(f'Processing data for {code}')
        out = processAirport(code, path)
        if final_result == None:
            final_result = out
        else:
            final_result = pd.concat([final_result,out])

    final_result.to_csv("../submission_csv/result.csv",index=False)



processData()



