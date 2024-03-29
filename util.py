from jax import random, jit
import tensorflow as tf
import neural_tangents as nt
import numpy.random as npr
import argparse
import jax.numpy as jnp
from logger import get_logger
logger = get_logger()

def args_parser():
    parser = argparse.ArgumentParser(description='Data Augmentation in Infinitely Wide Neural Networks')
    parser.add_argument('--model', default='resnet', choices=['fcn', 'cnn', 'resnet'],
                        help='the selection of models')
    parser.add_argument('--dataset', default='mnist10k', choices=['mnist', 'mnist10k'],
                        help='the selection of datasets')
    parser.add_argument("--epochs",  type=int, default=1, help="number of training steps")
    parser.add_argument("--batch-size",  type=int, default=0, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum mass")
    
    parser.add_argument("--num-inducing-points", type=int, default=750, help="number of inducing points")
    parser.add_argument("--select-method", default='random', choices=['random', 'first', 'greedy'], help="method for selecting inducing points")
    parser.add_argument("--device-count", type=int, default=-1, help="number of devices")
    
    parser.add_argument("--augment-X", type=str, default=None, help="the path to augmentation")
    parser.add_argument("--augment-y", type=str, default=None, help="the path to augmentation")
    
    args = parser.parse_args()
    logger.info(args)
    return args

def compute_num_batches(num_examples, batch_size):
    num_complete_batches, leftover = divmod(num_examples, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

def minibatch(x, y, batch_size, num_batches=None):
    if num_batches == None:
        num_batches = compute_num_batches(x.shape[0], batch_size)
        
    rng = npr.RandomState(0)
    num_train = x.shape[0]
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            batch_x, batch_y = x[batch_idx], y[batch_idx]
            # if padding:
            #     image_size = batch_x[0].shape
            #     pad_h = 224 - image_size[0]
            #     pad_w = 224 - image_size[1]
            #     # padding to 224x224
            #     pad_h = pad_h // 2 if pad_h > 0 else 0
            #     pad_w = pad_w // 2 if pad_w > 0 else 0
            #     batch_x = jnp.pad(batch_x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
                
            yield batch_x, batch_y

def create_checkpoint(model, output_dir: str, max_to_keep: int=5, **kwargs):
    ckpt = tf.train.Checkpoint(model=model, **kwargs)
    manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=max_to_keep)
    return ckpt, manager

def check_divisibility(train, test, batch_size, device_count):
    '''check if the number of data is divisible by batch size and device count'''
    train_num = train.shape[0]
    test_num = test.shape[0]
    if batch_size > 0:
        assert train_num % (batch_size * device_count) == 0, "training data size is not divisible by (batch size x device count)"
        assert test_num % batch_size == 0, "test data size is not divisible by batch size"

class PRNGKey():
    key = random.PRNGKey(0)

def init_random_state(seed: int):
    PRNGKey.key = random.PRNGKey(seed)

def split_key(num: int=2):
    keys = random.split(PRNGKey.key, num)
    PRNGKey.key = keys[0]
    return keys

def jit_and_batch(apply_fn, kernel_fn, batch_size, device_count, store_on_device):
    # JAX feature: compiles functions that they are executed as single calls to the GPU
    apply_fn = jit(apply_fn)
    kernel_fn = nt.batch(kernel_fn, batch_size, device_count, store_on_device)
    return apply_fn, kernel_fn
    
def softplus(X):
    return jnp.log(1 + jnp.exp(X))

def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)

def batch_kernel(kernel_fn, batch_size, x1, x2):
    if batch_size < 1:
        return kernel_fn(x1, x2, "nngp")
    
    N = x1.shape[0]
    M = x2.shape[0]
    kernel = []
    x1_start_indices = jnp.arange(0, N, batch_size)
    x1_end_indices = x1_start_indices + batch_size
    x2_start_indices = jnp.arange(0, M, batch_size)
    x2_end_indices = x2_start_indices + batch_size
    for x1_start, x1_end in zip(x1_start_indices, x1_end_indices):
        subkernel = []
        for x2_start, x2_end in zip(x2_start_indices, x2_end_indices):
            x1_end = min(x1_end, N)
            x2_end = min(x2_end, M)
            subkernel.append(kernel_fn(x1[x1_start:x1_end], x2[x2_start:x2_end], "nngp"))
        kernel.append(jnp.concatenate(subkernel, axis=1))
    kernel = jnp.concatenate(kernel)
    return kernel

def fill_diagonal(a, val):
    di = jnp.diag_indices(a.shape[0])
    return a.at[di].set(val)

def init_kernel_fn(model, stds, hyper_params):
    model = model(W_std=stds[0], b_std=stds[1], **hyper_params)
    kernel_fn = model.kernel_fn
    return kernel_fn

def kernel_diagonal(kernel_fn, data):
    input_shape = (1,) + data.shape[1:]
    N = data.shape[0]
    data = data.reshape((N, -1))
    kernel_wrapper = lambda x: kernel_fn(x.reshape(input_shape), None, "nngp")
    diagonal = jnp.apply_along_axis(kernel_wrapper, 1, data).flatten()
    return diagonal
    
if __name__ == "__main__":
    a = jnp.arange(25).reshape((5,5))
    print(a)
    print(fill_diagonal(a, 0))
    