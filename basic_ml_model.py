import numpy as np 
import pandas as pd 
import os


import mlflow
import mlflow.sklearn

import ssl

ssl._create_default_https_context = ssl._create_unverified_context



from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score

from sklearn.model_selection import train_test_split

import argparse

def get_data():
    URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    #read the data as dataframe
    try:
        df=pd.read_csv(URL, sep=';')
        return df
    except Exception as e:
        raise e


def evaluate(y_true,y_pred):
    '''mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    r_square = r2_score(y_true,y_pred)

    return mae,mse,rmse,r_square'''

    accuracy=accuracy_score(y_true,y_pred)

    return accuracy


def main(n_estimators,max_depth):
    #get data
    df=get_data()
    print(df)

    #train test split wit raw data
    train,test = train_test_split(df)
    x_train=train.drop(["quality"],axis=1)
    x_test=test.drop(["quality"],axis=1)
    y_train=train[["quality"]]
    y_test=test[["quality"]]
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    #model trainning
    '''lr=ElasticNet()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)'''

    rf =RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(x_train,y_train)
    pred=rf.predict(x_test)


    #evaluate the model
    #mae,mse,rmse,r2 = evaluate(y_test,pred)

    accuracy = evaluate(y_test,pred)

    #print(f"mean absolute error {mae}, mean squared error {mse}, root mean squared error {rmse}, r2_score {r2}")
    print(f"accuracy score {accuracy}")


if __name__ == "__main__":
    #getting inputs from command line for passing arguments
    args = argparse.ArgumentParser()
    #args.add_arguments(arg,default=str,help=str,type=str)
    #args.add_arguments(arg,default=str,help=str,type=str)
    args.add_argument("--n_estimators","-n",default=50,type=int)
    args.add_argument("--max_depth","-a",default=5,type=int)
    parse_args=args.parse_args()

    print(parse_args.n_estimators,parse_args.max_depth)
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e