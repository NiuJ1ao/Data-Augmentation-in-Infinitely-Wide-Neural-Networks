from jax import random
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
from keras.datasets import mnist
from util import init_random_state, split_key
from sklearn.model_selection import train_test_split
from examples import datasets
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

def synthetic_dataset2():
    def func(x):
        """Latent function."""
        return 1.0 * np.sin(x * 3 * np.pi) + \
            0.3 * np.cos(x * 9 * np.pi) + \
            0.5 * np.sin(x * 7 * np.pi)

    # Number of training examples
    n = 1000

    # Noise
    sigma_y = 0.2

    # Noisy training data
    X = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = func(X) + sigma_y * random.normal(random.PRNGKey(0), shape=(n, 1))

    # Test data
    X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
    f_true = func(X_test)
    
    return (X, y), (X_test, f_true)
    
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
    
    # x_train = x_train[:20000]
    # y_train = y_train[:20000]
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
        # # standardise images across channels
        # mean = np.mean(x, axis=(1,2), keepdims=True)
        # std = np.std(x, axis=(1,2), keepdims=True)
        # x = (x - mean) / std
    
    # normalise pixels of grayscale images
    x = x / np.float64(255.)
    
    # turn labels to one-hot encodings (-0.1 neg, 0.9 pos)
    if one_hot:
        y = np.eye(10)[y] - 0.1
        
    return x, y

def load_imdb(train_size, test_size, shuffle=True, 
              imdb_path='/tmp/imdb_reviews'):
    mask_constant = 100.
    
    x_train, y_train, x_test, y_test = datasets.get_dataset(
        name='imdb_reviews',
        n_train=train_size,
        n_test=test_size,
        do_flatten_and_normalize=False,
        data_dir=imdb_path,
        input_key='text')
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=shuffle)

    # Embed words and pad / truncate sentences to a fixed size.
    x_train, x_val, x_test = datasets.embed_glove(
        xs=[x_train, x_val, x_test],
        glove_path='/tmp/glove.6B.50d.txt',
        max_sentence_length=500,
        mask_constant=mask_constant)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == "__main__":
    pass
    