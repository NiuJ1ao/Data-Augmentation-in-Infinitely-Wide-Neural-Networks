from jax import random, jit
import numpy.random as npr
import argparse
from logger import get_logger
logger = get_logger()

def args_parser():
    parser = argparse.ArgumentParser(description='Data Augmentation in Infinitely Wide Neural Networks')
    parser.add_argument('--model', default='resnet', choices=['fcn', 'resfcn', 'resnet'],
                        help='an integer for the accumulator')
    parser.add_argument("--epochs",  type=int, default=1, help="number of training steps")
    parser.add_argument("--batch-size",  type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum mass")
    
    args = parser.parse_args()
    logger.info(args)
    return args

def minibatch(x, y, batch_size, num_batches):
    rng = npr.RandomState(0)
    num_train = x.shape[0]
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield x[batch_idx], y[batch_idx]

class PRNGKey():
    key = random.PRNGKey(0)

def init_random_state(seed: int):
    PRNGKey.key = random.PRNGKey(seed)

def split_key(num: int=2):
    keys = random.split(PRNGKey.key, num)
    PRNGKey.key = keys[0]
    return keys

def jit_fns(apply_fn, kernel_fn):
    # JAX feature: compiles functions that they are executed as single calls to the GPU
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return apply_fn, kernel_fn
    
if __name__ == "__main__":
    pass
    