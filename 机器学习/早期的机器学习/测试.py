from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=135)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False, stratify=y)


print(X_train.shape)
print(X_test.shape)
