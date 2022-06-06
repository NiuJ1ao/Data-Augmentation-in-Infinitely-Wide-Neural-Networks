from jax import random
import jax.numpy as np
from keras.datasets import mnist
from util import init_random_state, split_key
from sklearn.model_selection import train_test_split

def synthetic_dataset():
    init_random_state(10)
    
    train_points = 5
    test_points = 50
    noise_scale = 1e-1
    
    target_fn = lambda x: np.sin(x)
    
    _, x_key, y_key = split_key(3)
    
    train_xs = random.uniform(x_key, shape=(train_points, 1), minval=-np.pi, maxval=np.pi)
    train_ys = target_fn(train_xs)
    train_ys += noise_scale * random.normal(y_key, shape=train_xs.shape)
    train = (train_xs, train_ys)
    
    test_xs = np.linspace(-np.pi, np.pi, test_points).reshape(test_points, 1)
    test_ys = target_fn(test_xs)
    test = (test_xs, test_ys)
    
    return train, test
    
    
def load_mnist(shuffle=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    # train val split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=shuffle)
    
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    
    
