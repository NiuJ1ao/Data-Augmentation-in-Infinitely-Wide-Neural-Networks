from jax import random, jit
import jax.numpy as np
import argparse
from logger import logger

def args_parser():
    parser = argparse.ArgumentParser(description='Data Augmentation in Infinitely Wide Neural Networks')
    parser.add_argument('--model', default='resnet', choices=['fcn', 'resfcn', 'resnet'],
                        help='an integer for the accumulator')
    parser.add_argument("--epochs",  type=int, default=1, help="number of training steps")
    parser.add_argument("--batch-size",  type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    args = parser.parse_args()
    logger.info(args)
    return args

def minibatch(x, y, batch_size, train_epochs):
  """Generate minibatches of data for a set number of epochs."""
  epoch = 0
  start = 0
  key = random.PRNGKey(0)

  while epoch < train_epochs:
    end = start + batch_size

    if end > x.shape[0]:
      key, split = random.split(key)
      permutation = random.permutation(
          split,
          np.arange(x.shape[0], dtype=np.int32),
          independent=True
      )
      print(x.shape)
      print(permutation)
      assert False
      x = x[permutation]
      y = y[permutation]
      epoch += 1
      start = 0
      continue

    yield x[start:end], y[start:end]
    start = start + batch_size

class PRNGKey():
    key = random.PRNGKey(10)

def init_random_state(seed: int):
    PRNGKey.key = random.PRNGKey(seed)

def split_key(num: int=2) -> tuple:
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
    