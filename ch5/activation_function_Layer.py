import numpy as np

# TODO しっかり　これの計算をしてみること

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class ReluLayer:
    def __init__(self):
        self.mask = None
        
    def forward(self,x,y):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self,d_out):
        d_out[self.mask] = 0
        dx = d_out
        return dx
    

class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backword(self,d_out):
        dx = d_out * (1-self.out) * self.out
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
    

    