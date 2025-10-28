import numpy as np

# 活性化関数
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))  # 修正: np.exp(-x)

def identity_function(x):
    return x

def First_Layer():
    X = np.array([1.0, 0.5])  # 入力（2次元）
    W1 = np.array([[0.1, 0.3, 0.5],
                   [0.2, 0.4, 0.6]])  # 重み（2x3）
    B1 = np.array([0.1, 0.2, 0.3])   # バイアス（3次元）

    A = np.dot(X, W1) + B1
    Z = sigmoid_function(A)
    return Z

def Second_Layer(Y):
    W2 = np.array([[0.1,0.3],[0.2,0.6],[0.3,0.5]])
    B2 = np.array([0.1,0.2])

    A2 = np.dot(Y,W2) + B2
    Z2 = identity_function(A2)
    return Z2

X1, X2, X3 = First_Layer()
print(X1, X2, X3)

Y1,Y2 = Second_Layer(np.array([X1,X2,X3]))
print(Y1,Y2)


