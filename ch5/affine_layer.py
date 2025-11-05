# todo Affine Layerの計算をやってみること
import numpy as np
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
    
