#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 3/13/2022 10:15 PM
 @Author  : Shulu Chen
 @FileName: xgboost_main.py
 @Software: PyCharm
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import xgboost as xgb
from xgboost import plot_importance
import time
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
# df=pd.concat([X_data,y_data],axis=1)
# df_top=df[df['config']=="katl:D_8R_9L_A_10_8L_9R" or df['config']=="katl:D_26L_27R_A_26R_27L_28"]
# df_top=df[(df['config']=="katl:D_26L_27R_A_26R_27L_28")|(df['config']=="katl:D_8R_9L_A_10_8L_9R")]
X_train = X_train[['wind_direction']]
# y_train = df_top[['config']]
# X_train = X_train[['wind_direction','wind_speed','wind_gust','cloud_ceiling','visibility']]
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_train.rename('wind_direction','wind_speed','temperature','wind_gust','cloud_ceiling','visibility')
# for idx,x in enumerate(y_train['config']):
#     if '9' in x or "10" in x:
#         y_train['config'].iloc[idx] = "0"
#     if '26' in x:
#         y_train['config'].iloc[idx] = "1"
#     if 'other' in x:
#         y_train['config'].iloc[idx] = "2"
# print(y_train.config.unique())
le = LabelEncoder()
y_train = le.fit_transform(y_train.astype(str))

# Define XGBoost train function and Forecast function

#%%

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,train_size=0.7,random_state=1314)


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
model1=train_xgboost(X_train,y_train)
t2=time.time()-t1
plot_importance(model1)
plt.title('Seq1-Importance')
# plt.show()

# Forecast result


result1=forecast_data(X_test,model1)
print(classification_report(y_test, result1, target_names=le.classes_))
print("size of the training data:",len(y_train))
print("Training time: %.2f s"% t2)