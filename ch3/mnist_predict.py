import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
from Activation_Function import sigmoid_function
from Activation_Function import softmax

def get_data():
    (x_train,y_train),(x_test,y_test) = load_mnist(normalize=True, flatten=True,one_hot_label=False)
    return x_test,y_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,w1)+b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1,w2)+b2
    z2= sigmoid_function(a2)
    a3 = np.dot(z2,w3)+b3
    y = softmax(a3)

    return y

x,t = get_data()
network= init_network()

accuracy_cnt = 0

for i in range(len(x)):
    y= predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1 
    
print("正解率："+str(float(accuracy_cnt)/len(x)))