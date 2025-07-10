import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib
import math
import time

# 模型和标准化器保存路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'model', 'iris_logreg_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'src', 'model', 'iris_scaler.pkl')

def train(epochs=10, batch_size=16):

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 初始化新模型和标准化器
    model = SGDClassifier(loss='log_loss', penalty='l2', learning_rate='constant', eta0=0.01, max_iter=1, random_state=42)
    scaler = StandardScaler()
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 标准化
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 小批量训练
    n_samples = len(X_train)
    batches_per_epoch = math.ceil(n_samples / batch_size)
    
    for epoch in range(epochs):
        start_time = time.time()
        indices = np.random.permutation(n_samples)
        
        for batch in range(batches_per_epoch):
            # 获取小批量数据
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            model.partial_fit(X_batch, y_batch, classes=np.unique(y))
            
            # 每10个批次输出一次评估
            if batch % 10 == 0:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{batches_per_epoch}, 准确率: {acc:.4f}")
        
        # 每个轮次结束后评估并保存模型
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\nEpoch {epoch+1}/{epochs} 完成")
        print(f"测试集准确率: {acc:.4f}")
        print(f"耗时: {time.time() - start_time:.2f}秒\n")
        
        # 保存模型和标准化器
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"模型已保存至: {MODEL_PATH}")

def predict(sample):
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 标准化处理
    sample = np.array(sample)
    sample = sample.reshape(1, -1)
    sample = scaler.transform(sample)
    
    # 预测类别和概率
    predicted_class = model.predict(sample)
    predicted_proba = model.predict_proba(sample)
    
    return {
        '类别': predicted_class,
        '概率': predicted_proba
    }

if __name__ == "__main__":

    train(epochs=5, batch_size=16)
    
    sample = [5.1, 3.5, 1.4, 0.2] 
    result = predict(sample)
    
    print(f"预测示例: {sample}")
    print(f"预测类别: {result['类别']}")
    print(f"概率:{result['概率']}")
  