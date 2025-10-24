import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False,one_hot_label=True)
img = x_train[0]
label = t_train[0]
print("Label:", label)
img = img.reshape(28, 28)  # 1次元配列を28x28の2次元配列に変換
img_show(img)


# 画像をどのようにしてnumpyに変換しているか
print("Converting " + file_name + " to NumPy Array ...")
with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
data = data.reshape(-1, img_size)
print("Done")


# One_hot labelをどのように実装しているか



