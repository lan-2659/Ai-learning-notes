from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor,Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import os
import joblib
import math
# 如果模型在本地有就加载出来继续训练(因为保存的模型其实就是曾经训练修改过很多次的w和b)
# 如果没有 就创建 一个模型 第一次训练
# print()
mode_path=os.path.join(os.path.dirname(__file__),'src',"model","lasso_regressor_model.pkl")
transfer_path=os.path.join(os.path.dirname(__file__),'src',"model","lasso_transfer.pkl")
# print(mode_path)
# print(os.path.exists(mode_path))
def train():
    # 1.加载模型
    model=None
    transfer=None
    if os.path.exists(mode_path):
        model=joblib.load(mode_path)
        transfer=joblib.load(transfer_path)
    else:
        model=Lasso(fit_intercept=False,max_iter=100,warm_start=True,alpha=1,tol=0.001)
        transfer=StandardScaler()
    # 2.加载数据集
    x,y=fetch_california_housing(data_home="./src",return_X_y=True)
    # 3.数据集划分
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
    # 4.标准化
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    # 5.训练模型
    epochs=100
    batch_size=32
    for epoch in range(epochs):
        indexs=np.arange(len(x_train))
        np.random.shuffle(indexs)
        # print(indexs,"==========")
        # exit()#退出python程序
        start_time=time.time()
        for batch in range(math.ceil(len(x_train)/batch_size)):
            start=batch*batch_size
            end=min((batch+1)*batch_size,len(x_train))
            index=indexs[start:end]
            x_batch=x_train[index]#32条数据
            y_batch=y_train[index]
            # 训练这32条数据(本来应该用32条数据计算损失函数 然后算梯度 再做梯度下降)
            model.fit(x_batch,y_batch)
        y_pred=model.predict(x_test)
        e=mean_squared_error(y_test,y_pred)
        # end_time=time.time()
        score=model.score(x_test,y_test)
        print(f"训练轮次:{epoch}/{epochs} 均方误差为：{e} score:{score} time:{time.time()-start_time}")
        # 保存模型
        joblib.dump(model,mode_path)
        joblib.dump(transfer,transfer_path)

def detect():
    model=joblib.load(mode_path)
    transfer=joblib.load(transfer_path)
    x_true=[[10,20,10,2,2,1,1,9]]
    x_true=transfer.transform(x_true)
    print(model.predict(x_true))       
if __name__ == '__main__':
    train()
    # detect()