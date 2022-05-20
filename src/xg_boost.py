#!/usr/bin/python3
# coding: utf-8

import matplotlib.pyplot as plt
import time
import traceback
import warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import plot_importance
warnings.filterwarnings('ignore')

AIRPORTS = ['katl','kclt','kden','kdfw', 'kjfk', 'kmem', 'kmia', 'kord', 'kphx', 'ksea']
# AIRPORTS=['kden','ksea']


PARAMS = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        "num_class": 26,
        'gamma': 0.01,
        'max_depth': 6,
        'lambda': 1,
        'subsample': 0.99,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 0,
        'eta': 0.01,
        'seed': 100,
        # 'tree_method': 'gpu_hist' ,
    }




def runXgBoost(test_size = 0.33, epochs = 3000, save = False, evaluate = True):
    airport_accuracy = {}
    for airport in tqdm(AIRPORTS):
        print(f"Running {airport}")
        Data = pd.read_csv(f'../out/{airport}_x_train.csv')

        ## Full features
        X_data = Data[['year','month','day','hour','quarter','weekofyear',
                          'dayofweek','weekend','hour_section','wind_direction','wind_speed',
                          'temperature','wind_gust','cloud_ceiling','visibility']]
        ## Only weather
        # X_data = Data[['wind_direction','wind_speed',
        #                   'temperature','wind_gust','cloud_ceiling','visibility']]
        ## Only time
        # X_data = Data[['year','month','day','hour','quarter','weekofyear',
        #                'dayofweek','weekend','hour_section']]
        y_data = Data[['config']].to_numpy().ravel()

        le = LabelEncoder()
        y_data = le.fit_transform(y_data.astype(str))

        X_train,X_test,y_train,y_test = train_test_split(X_data, y_data, test_size=test_size,random_state=1314)

        PARAMS['num_class'] = len(le.classes_)

        try:
            start= time.time()
            model = trainModel(X_train, y_train, epochs, save, airport)
            end=time.time()-start

            if evaluate:
                airport_accuracy[airport] = {'classes':len(le.classes_), 'acc':round(evaluateModel(X_test, y_test, model, le.classes_),6)}

            print("Size of the training data:",len(y_train))
            print("Number of classes evaluated:", len(le.classes_))
            print("Training time: %.2f s"% end)
        except xgb.core.XGBoostError as e :
            print(f'+++++++++++++++++++++++++++++++++++\nERROR: Error in {airport}\n+++++++++++++++++++++++++++++++++++')
            traceback.print_exc()
            print()
    # print(airport_accuracy)
    for key,value in airport_accuracy.items():
        print(f'{key}:{value}')


def trainModel(X_train, y_train, epochs = 100, save = False, airport = None):
    print(y_train.shape)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    plst = list(PARAMS.items())
    model = xgb.train(plst, dtrain, epochs)
    plot_importance(model)
    plt.title(f'Importance of {airport}')
    plt.show()
    if save:
        model.save_model(f'models/{airport}_seq1.model')
        model.dump_model(f'models/dump/{airport}seq1_model.txt')

    return model

def evaluateModel(X_test, y_test, model, classes):
    pred = predict(X_test, model)
    c_report = classification_report(y_test, pred, target_names=classes)
    # print(c_report)
    return accuracy_score(y_test, pred)

def predict(X_vals, model):
    dtest = xgb.DMatrix(X_vals)
    return model.predict(dtest)
start_total=time.time()
runXgBoost()
end_total=time.time()-start_total
print(f"total running time: %.2f s" % end_total)