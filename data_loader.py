from jax import random
import jax.numpy as np
from keras.datasets import mnist
from util import init_random_state, split_key
from sklearn.model_selection import train_test_split
from logger import get_logger
logger = get_logger()

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
    
def load_mnist(shuffle: bool=True, flatten: bool=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28) 
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    x_train, y_train = preprocess_mnist(x_train, y_train, flatten)
    x_test, y_test = preprocess_mnist(x_test, y_test, flatten)
    
    # train val split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=shuffle)
    
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_val = x_val[:1000]
    y_val = y_val[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    logger.info(f"MNIST: {len(x_train)} train, {len(x_val)} val, {len(x_test)} test samples.")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_mnist(x, y, flatten: bool=False):
    if flatten:
        x = x.reshape(x.shape[0], -1)
    else:
        # reshape to have single channel
        x = x.reshape(x.shape + (1,))
    
    # normalise pixels of grayscale images
    x /= np.float32(255.)
    
    # turn labels to one-hot encodings (-0.1 neg, 0.9 pos)
    y = np.eye(10)[y] - 0.1
        
    return x, y
    
if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(flatten=True)
    
    