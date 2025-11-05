import numpy as np
import sys,os
sys.path.append(os.pardir)
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/common/")
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/dataset")
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/ch7/")
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/ch5/")
from mnist import load_mnist
from convolution import Convolution,Pooling
from collections import OrderedDict
from Two_Layer_net_with_Layer import * 
from activation_function_Layer import *

# 画像をテストデータと訓練データに分ける
# ハイパーパラメータの定義
    # 重みの初期化
    # レイヤーの生成
    
# ミニバッチを作る
# 重みを更新する
# 結果を格納

# 画像をテストデータと訓練データに分ける


class SimpleConvNet:

    def __init__(self,input_dim=(1,28,28),
                 conv_param = {'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                 hidden_size = 100, output_size = 10, weight_init_std = 0.01):
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params ={}

        # 重みの設定
        self.params['W1'] = np.random.randn(filter_num, input_dim[0], filter_size, filter_size) * weight_init_std
        self.params['B1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size)
        self.params['B2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['B3'] = np.zeros(output_size)

        # レイヤーの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params["B1"],conv_param['stride'],conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
    
    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['B1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['B2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['B3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads 
    
        