import numpy as np
import matplotlib.pylab as plt
import argparse


def step_function(x):
    return np.array(x > 0, dtype=int)

def relu(x):
    return np.maximum(0, x)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def plot_function(func_name):
    x = np.arange(-5, 5, 0.1)

    if func_name == "step":
        y = step_function(x)
    elif func_name == "relu":
        y = relu(x)
    elif func_name == "sigmoid":
        y = sigmoid_function(x)
    else:
        raise ValueError(f"Unknown function name: {func_name}")

    plt.plot(x, y)
    plt.title(f"{func_name} function")
    plt.ylim(-0.1, 1.1 if func_name != "relu" else np.max(y) + 0.5)
    plt.grid(True)
    plt.show()


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot activation functions (step, relu, sigmoid)"
    )
    parser.add_argument(
        "--func",
        type=str,
        default="step",
        choices=["step", "relu", "sigmoid"],
        help="Choose which function to plot",
    )
    args = parser.parse_args()

    plot_function(args.func)
