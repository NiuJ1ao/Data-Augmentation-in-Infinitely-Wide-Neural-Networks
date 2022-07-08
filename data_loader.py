from jax import random
import jax.numpy as np
from jax.config import config
from matplotlib import image
config.update("jax_enable_x64", True)
from keras.datasets import mnist
from util import init_random_state, split_key
from sklearn.model_selection import train_test_split
from logger import get_logger
logger = get_logger()

def synthetic_dataset():
    config.update("jax_enable_x64", False)
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
    
def load_mnist(shuffle: bool=True, 
               flatten: bool=False, 
               one_hot: bool=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    assert x_train.shape == (60000, 28, 28) 
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    # train val split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=shuffle)
    
    x_train, y_train = preprocess_mnist(x_train, y_train, flatten, one_hot)
    x_val, y_val = preprocess_mnist(x_val, y_val, flatten, one_hot)
    x_test, y_test = preprocess_mnist(x_test, y_test, flatten, one_hot)
    
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    # x_val = x_val[:1000]
    # y_val = y_val[:1000]
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]
    
    logger.info(f"MNIST: {x_train.shape} train, {x_val.shape} val, {x_test.shape} test samples.")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_mnist(x, y, flatten: bool=False, one_hot: bool=True):
    if flatten:
        x = x.reshape(x.shape[0], -1)
    else:
        # reshape to have single channel for CNN
        x = x.reshape(x.shape + (1,))
        # image_size = x[0].shape
        # pad_h = 224 - image_size[0]
        # pad_w = 224 - image_size[1]
        # # padding to 224x224
        # pad_h = pad_h // 2 if pad_h > 0 else 0
        # pad_w = pad_w // 2 if pad_w > 0 else 0
        # x = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    
    # normalise pixels of grayscale images
    x = x / np.float64(255.)
    
    # turn labels to one-hot encodings (-0.1 neg, 0.9 pos)
    if one_hot:
        y = np.eye(10)[y] - 0.1
        
    return x, y
    
if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(flatten=True)
    
    