import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
print("Loading MNIST dataset...")
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)
# Normalize: ピクセル値を0.0~1.0に正規化
# Flatten: 画像を一次元配列に平にする  
# One-hot label: ラベルをone-hot配列に変換

print("MNIST dataset loaded.")
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
