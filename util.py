from jax import random, jit
import tensorflow as tf
import neural_tangents as nt
import numpy.random as npr
import argparse
from logger import get_logger
logger = get_logger()

def args_parser():
    parser = argparse.ArgumentParser(description='Data Augmentation in Infinitely Wide Neural Networks')
    parser.add_argument('--model', default='resnet', choices=['fcn', 'resfcn', 'resnet'],
                        help='an integer for the accumulator')
    parser.add_argument("--epochs",  type=int, default=1, help="number of training steps")
    parser.add_argument("--batch-size",  type=int, default=0, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum mass")
    
    parser.add_argument("--num-inducing-points", type=int, default=750, help="number of inducing points")
    
    parser.add_argument("--device-count", type=int, default=-1, help="number of devices")
    
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
            yield x[batch_idx], y[batch_idx]

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
    
if __name__ == "__main__":
    pass
    