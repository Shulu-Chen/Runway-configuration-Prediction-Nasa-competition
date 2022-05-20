from tqdm import tqdm as tqdm
from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/train_features")
airport_directories = sorted(path for path in DATA_PATH.glob("k*"))

airport_directory = airport_directories[0]
data_files = list(airport_directory.glob("*"))


# lamp weather data
airport_code = airport_directory.name
filename = f"{airport_code}_lamp.csv.bz2"
filepath = airport_directory / filename

airport_lamp = pd.read_csv(filepath, parse_dates=["timestamp"])
airport_lamp=airport_lamp.sort_values('timestamp')
airport_lamp=airport_lamp.drop_duplicates()

# check and impute missing value
airport_lamp.info()
airport_lamp[airport_lamp['temperature'].isna()]['timestamp'].unique()
# airport_lamp[airport_lamp['timestamp']=='2020-12-11 19:30:00']
# airport_lamp[airport_lamp['forecast_timestamp']=='2021-07-28T15:00:00']

lamp_nan = airport_lamp[airport_lamp['temperature'].isna()]
lamp_nan_index = airport_lamp[airport_lamp['temperature'].isna()].index
for i in lamp_nan_index:
    forecast_timestamp = airport_lamp.loc[i]['forecast_timestamp']
    airport_lamp.loc[i]= airport_lamp[airport_lamp['forecast_timestamp']==forecast_timestamp].iloc[-2]


lamp_nan_index = airport_lamp[airport_lamp['temperature'].isna()].index
for i in lamp_nan_index:
    forecast_timestamp = airport_lamp.loc[i]['forecast_timestamp']
    airport_lamp.loc[i]= airport_lamp[airport_lamp['forecast_timestamp']==forecast_timestamp].iloc[-3]


lamp_nan_index = airport_lamp[airport_lamp['temperature'].isna()].index
for i in lamp_nan_index:
    forecast_timestamp = airport_lamp.loc[i]['forecast_timestamp']
    airport_lamp.loc[i]= airport_lamp[airport_lamp['forecast_timestamp']==forecast_timestamp].iloc[-4]

airport_lamp[airport_lamp['temperature'].isna()]


# train label

DATA_PATH = Path("data/open_train_labels")
open_train_labels = pd.read_csv(DATA_PATH / "open_train_labels.csv.bz2", parse_dates=["timestamp"])

labels= open_train_labels[(open_train_labels['active']==1) &(open_train_labels['airport']=='katl')]



MAXLOOKAHEAD = 360
min_label_time, max_label_time = min(pd.to_datetime(labels["timestamp"])),max(pd.to_datetime(labels["timestamp"]))
    # data_files = list(path.glob("*"))
    #
    # weather_file = f"{code}_lamp.csv.bz2"
    # weather_path = path / weather_file

    # airport_weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
airport_weather = airport_lamp[(pd.to_datetime(airport_lamp["timestamp"]) >= min_label_time-pd.DateOffset(hours=1))&(pd.to_datetime(airport_lamp["timestamp"]) <= max_label_time)]


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


OUT_PATH = Path("out")
X_train.to_csv(OUT_PATH / f'test_katl_timebased_x_train.csv', index=False)
y_train.to_csv(OUT_PATH / f'test_katl_timebased_y_train.csv', index=False)


