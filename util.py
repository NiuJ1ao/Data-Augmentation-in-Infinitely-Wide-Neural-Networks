from jax import random
from jax import jit
import argparse
from logger import logger

def args_parser():
    parser = argparse.ArgumentParser(description='Data Augmentation in Infinitely Wide Neural Networks')
    parser.add_argument('--model', default='resnet', choices=['fcn', 'resfcn', 'resnet'],
                        help='an integer for the accumulator')
    parser.add_argument("--training-steps", type=int, default=10000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    
    args = parser.parse_args()
    logger.info(args)
    return args

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
    