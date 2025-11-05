
import numpy as np
import sys, os
sys.path.append("/Users/kenijkaminogo/Desktop/IT学習/Machine_Learning/deeplearning_from_scratch/DeepLearningFromScratch_01/ch3/")
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/ch3/")
sys.path.append("/Users/kenijkaminogo/Desktop/IT学習/Machine_Learning/deeplearning_from_scratch/DeepLearningFromScratch_01/ch4/")
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/ch4")
# from gradient import numerical_gradient 
from Activation_Function import *
from loss_function import *
from collections import OrderedDict

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b

        return out

    def backward (self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.dB = np.sum(dout,axis=0)

        return dx

class soft_max_with_loss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self,y,t):
        self.y = y
        self.t = t
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,d_out=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size

        return dx



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
        # --------------------追加範囲-------------------------------
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['B1'])
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['B2'])
        self.last_Layer = soft_max_with_loss()

    def predict(self,x):
        # --------------------過去範囲-------------------------------
        # W1,W2 = self.params['W1'],self.params['W2']
        # B1,B2 = self.params['B1'],self.params['B2']

        # A1 = np.dot(x,W1) + B1
        # Z1 = sigmoid_function(A1)

        # A2 = np.dot(Z1,W2) + B2
        # Z2 = sigmoid_function(A2)

        # y = softmax(Z2)
        # return y
        for layer in self.layers.values():
            x = layer.forward(x)
        return x 
    
    def loss(self,x,t):
        y = self.predict(x)
        return self.last_Layer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim!=1 : t = np.argmax(t,axis=1)

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
    
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self.last_Layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] =self.layers['Affine1'].dW
        grads['B1'] =self.layers['Affine1'].dB
        grads['W2'] =self.layers['Affine2'].dW
        grads['B2'] =self.layers['Affine2'].dB
        
        return grads
    
    