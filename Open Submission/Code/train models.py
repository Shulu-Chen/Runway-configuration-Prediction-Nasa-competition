import warnings

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.externals import joblib
warnings.filterwarnings('ignore')

params = {
        'booster': 'gbtree',
        # 'objective': 'multi:softmax',
        'objective': 'multi:softprob',
        "num_class": None,
        'gamma': 0.01,
        'max_depth': 8,
        'lambda': 1,
        'subsample': 0.99,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.01,
        'seed': 100,
        # 'tree_method': 'gpu_hist' ,

    }

AIRPORTS = ['katl', 'kclt', 'kden', 'kdfw', 'kjfk', 'kmem', 'kmia','kord', 'kphx', 'ksea']
TRAINPATH = "../data/train_features"
X_files = "_timebased_x_train2.csv"
y_files = "_timebased_y_train2.csv"

reload_model = False


def main():

    for airport in AIRPORTS:
        X_location = f'../out/{airport}{X_files}'
        y_location = f'../out/{airport}{y_files}'

        airport_configs = pd.read_csv(f'{TRAINPATH}/{airport}/{airport}_airport_config.csv.bz2')

        X, y = pd.read_csv(X_location), pd.read_csv(y_location)
        unique_labels = y['config'].unique()
        params["num_class"] = len(unique_labels)
        print(f'Found {params["num_class"]} unique labels for {airport.upper()}')
        le = LabelEncoder()
        le.fit(unique_labels)

        len_unique_times = len(y['label_timestamp'].unique())



        #Random shuffle
        # X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=1314)
        # y_train = le.transform(y_train['config'])
        # y_test = le.transform(y_test['config'])

        #Whole data as training set
        X_train = X
        y_train = le.transform(y['config'])

        # Generate test train and validation sets
        # train_end = int(np.ceil(len_unique_times*0.7)*12)
        # X_train, y_train = X.loc[:train_end].copy(), le.transform(y.loc[:train_end].copy()['config'])
        # X_test, y_test = X.loc[train_end:].copy(),  le.transform(y.loc[train_end:].copy()['config'])

        X_train = prepare_train(X_train.to_numpy(), y_train, airport_configs, airport, le)[:,1:]

        #Save and reload models
        model_dir = f"../Models/{airport}_model.pkl"

        if reload_model and os.path.exists(model_dir):
            model = joblib.load(model_dir)
            print(f"Reload {airport} model successfully")
        else:
            model = train_model(X_train, y_train)
            joblib.dump(model,model_dir)
            print(f"Save {airport} model successfully")

        # run_test(X_test.to_numpy(), y_test, airport_configs, model, le, airport, unique_labels)
        
def prepare_train(X_train, y_train, actual_configs, airport, encoder):

    corrected_X = []

    for i in tqdm(range(len(X_train))):
        if i% 12 == 0:
            time = pd.to_datetime(X_train[i,0])
            encoded = get_real_config(actual_configs, encoder, airport, time)
            corrected_X.append(np.append(X_train[i], encoded))
        else:
            corrected_X.append(np.append(X_train[i].copy(), y_train[i-1]))
    
    return np.array(corrected_X)


def run_test(X, y, airport_configs, model, encoder, airport, labels):
    pre_config = 0
    true, pred, pred_prob = [], [], []
    M = params["num_class"]
    N = len(X)
    for i in tqdm(range(N)):
        X_val = X[i]
        if X_val[1] == 30:
            time = pd.to_datetime(X_val[0])
            pre_config = get_real_config(airport_configs, encoder, airport, time)
        
        X_val = np.array([np.append(X_val, pre_config)[1:]])

        est_prob = make_prediction(X_val, model)[0]
        true.append(y[i])
        pred_prob.append(est_prob)
        # pre_config=encoder.transform([labels[np.argmax(est_prob)]])
        pre_config=np.argmax(est_prob)
        # print(pre_config,y[i],est_prob)
        pred.append(pre_config)


    
    true = np.array(true)
    pred = np.array(pred)
    a_score = accuracy_score(true,pred)
    ll_score = log_loss(true,pred_prob)
    print(f'{airport.upper()}: Loss = {ll_score} | Accuracy = {a_score:.3f}')

    # log_loss(group["active"], predictions.loc[group.index, "active"])
    # for airport, group in training_labels.groupby("airport")]

def get_real_config(airport_configs, encoder, airport, time):
    valid = airport_configs[(pd.to_datetime(airport_configs["timestamp"]) <= time)].to_numpy()[-1,1]
    valid = f'{airport}:{valid}'
    try:
        encoded = encoder.transform([valid])[0]
    except:
        encoded = encoder.transform([f'{airport}:other'])[0]
    
    return encoded


def train_model(X,y):
    dtrain = xgb.DMatrix(X, label=y)
    num_rounds = 3000
    plst = list(params.items())

    return xgb.train(plst, dtrain, num_rounds)

def make_prediction(X, model):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)

main()
