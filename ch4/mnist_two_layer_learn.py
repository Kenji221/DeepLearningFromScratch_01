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
iters_num = 50s
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
