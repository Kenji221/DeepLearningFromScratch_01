
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


import sys,os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
print("Loading MNIST dataset...")
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)   
import  matplotlib as plt
from tqdm import tqdm

from mnist_two_layer import TwoLayerNet

# print確認
print("x train size will be ",x_train.shape)
print("t train size will be ",t_train.shape)

#　ハイパーパラメータの定義
learn_rate = 0.01
iters_num = 50
train_size = x_train.shape[0]
print("the train size will be described as ",train_size)
batch_size = 100

network = TwoLayerNet(input_size = 784,hidden_size=50,output_size=10)

loss_values = []

#　ミニバッチを何回回すか
for i in tqdm(range(iters_num), desc="Training"):
    # ミニバッチ用データ取得
    batch_mask = np.random.choice(train_size,batch_size)
    print(batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print("x_batch size will be ")
    print(x_batch.shape)
    print("t_batch size will be ")
    print(t_batch.shape)

    # 勾配の計算
    grad = network.numerical_gradient(x_batch,t_batch)

    # パラメータの更新
    for key in ('W1','W2','B1','B2'):
        network.params[key] -= grad[key]*learn_rate
    
    loss = network.loss(x_batch,t_batch)
    loss_values.append(loss)

for key in ('W1','W2','B1','B2'):
    print(network.params[key])


import matplotlib.pyplot as plt

# loss_values の描画
plt.plot(loss_values)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()