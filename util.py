from jax import random
from jax import jit

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
    