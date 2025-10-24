#　勾配について
## 微分

import sys, os
sys.path.append(os.pardir)
import numpy as np

def numerical_diff(f,x):
    h = 1e-4 
    return (f(x+h)-f(x-h))/(h*2)

### 微分具体例
def function_1(x):
    return 0.01*x**2 + 0.1*x

#---- 5で微分
print(numerical_diff(function_1,5))


## 3変数の２次式
def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


# 勾配降下
# f: 関数 / x: 初期値 / lr:学習率 / step_num: ステップ数
def gradient_descent(f,init_x, lr = 0.1,step_num=100):
    x = init_x

    for i in range(step_num):
        grad= numerical_gradient(f,x)
        x -= grad*lr

    return x

init_x = np.array([-3.0,4.0])
min = gradient_descent(function_2,init_x)
print(min)





