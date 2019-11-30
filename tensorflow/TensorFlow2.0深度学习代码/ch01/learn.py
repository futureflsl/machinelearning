import tensorflow as tf
import  matplotlib
from    matplotlib import pyplot as plt
import numpy as np
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiti']
matplotlib.rcParams['axes.unicode_minus']=False


if __name__ == '__main__':
    x  = tf.constant(4.)
    w  = tf.constant(6.)
    b  = tf.constant(7.)
    # add
    print(x+w)

    # autogradient
    with tf.GradientTape() as tape:
        tape.watch([w])
        y = w * x + b

    [dy_dw] = tape.gradient(y,[w])
    print(dy_dw)
    x = [10 ** i for i in range(9)]
    cpu_data = tf.random.normal([9])
    gpu_data = tf.random.normal([9])
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()

    data = {'a': np.arange(50),
    'c': np.random.randint(0, 50, 50),
    'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('entry a')
    plt.ylabel('entry b')
    plt.legend
    plt.show()