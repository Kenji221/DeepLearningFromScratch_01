import sys,os
sys.path.append(os.pardir)
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/ch3/")
import numpy as np
from loss_function import cross_entropy_error
from Activation_Function import softmax
from gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss
    
net = SimpleNet()
print(net.W)


x = np.array([0.6,0.9])
p = net.predict(x)

t = np.array([0,0,1])
print(net.loss(x,t))

