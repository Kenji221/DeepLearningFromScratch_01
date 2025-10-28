import numpy as np
import sys, os
sys.path.append("/Users/kenijkaminogo/Desktop/IT学習/Machine_Learning/deeplearning_from_scratch/DeepLearningFromScratch_01/common")
from gradient import numerical_gradient 
from Activation_Function import *
from loss_function import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):
        # 重みの初期化
        ## 初期メソッドのため最初にのみ生成される乱数となっている
        ## 後続では実際の初期についても触れる
        self.params = {}
        self.params['W1'] = weight_init_std* np.random.randn(input_size,hidden_size)
        self.params['B1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['B2'] = np.zeros(output_size)

    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        B1,B2 = self.params['B1'],self.params['B2']

        A1 = np.dot(x,W1) + B1
        Z1 = sigmoid_function(A1)

        A2 = np.dot(Z1,W2) + B2
        Z2 = sigmoid_function(A2)

        y = softmax(Z2)
        return y


    def loss(self,x,t):
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = (np.sum(y==t)) / (float(x.shape[0]))
        return accuracy


    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}

        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['B1'] = numerical_gradient(loss_W,self.params['B1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['B2'] = numerical_gradient(loss_W,self.params['B2'])

        return grads

