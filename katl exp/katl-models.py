
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier

# =========================================================== read in data ============================================
X_train = pd.read_csv('out/test_katl_timebased_x_train.csv')
y_train = pd.read_csv('out/test_katl_timebased_y_train.csv')


# =========================================================== add features  ============================================

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


features_new, colNames_new = timeSeriesCreation(X_train['label_timestamp'],precision_high=True)

seq = ['month','day','hour','weekofyear','dayofweek']
features_new1 = features_new[seq]

X_train = pd.concat([X_train,features_new1],axis=1,join='outer')

# ============================================================== actucal label ====================================================

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/train_features")
start_time_stamp = "2020-11-06T23:00:00"
airport = "katl"

airport_directories = sorted(path for path in DATA_PATH.glob("k*"))

airport_directory = airport_directories[0]
data_files = list(airport_directory.glob("*"))

airport_directory = airport_directories[0]

airport_code = airport_directory.name
filename = f"{airport_code}_airport_config.csv.bz2"
filepath = airport_directory / filename

airport_config = pd.read_csv(filepath, parse_dates=["timestamp"])




# Enlabel actual runway config to train label

y_train_unique = np.unique(y_train['config'])
for i in range(0,len(y_train_unique)):
    y_train_unique[i]=y_train_unique[i].replace('katl:','')  # train_label_unique is open training label

# select 'other' config
other = np.setdiff1d(airport_config['airport_config'],y_train_unique)
for idx,x in enumerate(airport_config['airport_config']):
    if x in other:
        airport_config['airport_config'].iloc[idx] = 'other'










# for 30mins lookahead - actual config feature
act_config = []
for i in range(len(X_train[X_train['lookahead'] == 30])):
    act_config.append(airport_config[y_train[y_train['lookahead']==30]['label_timestamp'].iloc[i] > airport_config['timestamp']]
                                   .iloc[-1]['airport_config'])

act_config=pd.DataFrame(act_config)
act_config.columns = ['act_config']


# X_train.drop(columns='act_config')

# add one column named 'act_config'
X_train['act_config']=0
for i in range(len(act_config)):
    X_train['act_config'][12*i] = act_config['act_config'].iloc[i]


#  train model for every 30 mins

X_train_30 = X_train[X_train['lookahead']==30][['wind_direction', 'wind_speed', 'temperature', 'month', 'day', 'hour','weekofyear', 'dayofweek','act_config']]
y_train_30 = y_train[y_train['lookahead']==30]['config']


le = LabelEncoder()
y_train_30 = le.fit_transform(y_train_30.astype(str))
X_train_30['act_config'] = le.fit_transform(X_train_30['act_config'])
Xtrain,Xtest,ytrain,ytest = train_test_split(X_train_30,y_train_30, train_size=0.7,random_state=1314)



def train_xgboost(Data,Label):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        "num_class": 26,
        'gamma': 0.01,
        'max_depth': 6,
        'lambda': 1,
        'subsample': 0.99,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.01,
        'seed': 100,
        # 'tree_method': 'gpu_hist' ,

    }
    dtrain = xgb.DMatrix(Data, label=Label)
    num_rounds = 300
    plst = list(params.items())
    model = xgb.train(plst, dtrain, num_rounds)
    # model.save_model('seq1.model')
    # model.dump_model('seq1_model.txt')
    return model

def forecast_data(Data,Model):
    dtest = xgb.DMatrix(Data)
    ans = Model.predict(dtest)
    return ans


# Train XGBoost Model
t1=time.time()
model1=train_xgboost(Xtrain,ytrain)
t2=time.time()-t1
plot_importance(model1)
plt.title('Seq1-Importance')
plt.show()

# Forecast result
result1=forecast_data(Xtest,model1)
print(classification_report(ytest, result1, target_names=le.classes_))
print("size of the training data:",len(y_train))
print("Training time: %.2f s"% t2)


# creat 30 mins predicted config
model30=train_xgboost(X_train_30,y_train_30)
config=forecast_data(X_train_30,model30)



timestamp = [60,90,120,150,180,210,240,270,300,330,360]
for i in timestamp:
    X_train_sub = X_train[X_train['lookahead']==i][['wind_direction', 'wind_speed', 'temperature', 'month', 'day', 'hour','weekofyear', 'dayofweek','act_config']]
    X_train_sub['pre_predict_config']= config
    X_train_sub = X_train_sub.drop(columns = 'act_config')
    y_train_sub = y_train[y_train['lookahead']== i]['config']


    le = LabelEncoder()
    y_train_sub = le.fit_transform(y_train_sub.astype(str))


    Xtrain,Xtest,ytrain,ytest = train_test_split(X_train_sub,y_train_sub, train_size=0.7,random_state=1314)

    # Train XGBoost Model
    t1=time.time()
    model=train_xgboost(Xtrain,ytrain)
    t2=time.time()-t1


    plot_importance(model)
    plt.title(f'Feature-Importance lookahead{i} mins')
    plt.show()

    # Forecast result
    result=forecast_data(Xtest,model)
    print(f'The runway config prediction result of lookahead {i} mins.')
    print(classification_report(ytest, result, target_names=le.classes_))
    print("size of the training data:",len(ytrain))
    print("Training time: %.2f s"% t2)


    # creat 30 mins predicted config
    model_new =train_xgboost(X_train_sub,y_train_sub)
    config = forecast_data(X_train_sub,model_new)