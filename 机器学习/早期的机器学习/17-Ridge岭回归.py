from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import os
import joblib
import math


def train():
    x,y=fetch_california_housing(data_home="./src",return_X_y=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    model=Ridge(alpha=1,fit_intercept=True,max_iter=100)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    e=mean_squared_error(y_test,y_pred)
    print(f"均方误差为：{e}")
    print(f"模型参数为：{model.coef_}")
    print(f"模型偏置为：{model.intercept_}")

if __name__=="__main__":
    train()