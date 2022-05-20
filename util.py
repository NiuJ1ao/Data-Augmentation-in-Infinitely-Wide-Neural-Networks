from jax import random
from jax import jit

class PRNGKey():
    key = random.PRNGKey(10)

def init_random_state(seed: int):
    PRNGKey.key = random.PRNGKey(seed)

def split_key(num: int=2) -> tuple:
    PRNGKey.key, *keys = random.split(PRNGKey.key, num)
    return PRNGKey.key, *keys

def jit_fns(apply_fn, kernel_fn):
    # JAX feature: compiles functions that they are executed as single calls to the GPU
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return apply_fn, kernel_fn
    
if __name__ == "__main__":
    print(PRNGKey.key)
    k1, k2 = split_key()
    print(k1, k2)
    print(PRNGKey.key)
    k1, k2, k3 = split_key(3)
    print(k1, k2)
    print(PRNGKey.key)
    