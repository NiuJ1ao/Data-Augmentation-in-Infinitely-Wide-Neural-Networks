import numpy as np
import jax.numpy as jnp
from jax import lax
from logger import get_logger
logger = get_logger()

def random_select(X, M):
    if M > X.shape[0]:
        logger.warning(f"M is larger than the number of samples, M = {M}")
        return X
    return X[np.random.choice(X.shape[0], M, replace=False)]

def first_n(X, M):
    return X[:M, :]

def greedy_variance(X, M, kernel_fn, sample = False, threshold = 0.0):
    """Jax implementation of greedy variance selection 
    https://github.com/markvdw/RobustGP/blame/0819bc9370f8e974f7f751143224d59d990e9531/robustgp/init_methods/methods.py#L107


    Args:
        X (_type_): _description_
        M (_type_): _description_
        kernel_fn (_type_): _description_
        sample (bool, optional): _description_. Defaults to False.
        threshold (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    N = X.shape[0]
    perm = np.random.permutation(N)
    training_inputs = X[perm]
    indices = jnp.zeros(M, dtype=int) + N
    # diagnoal of kernel
    lambda_kernel = lambda x: kernel_fn(x.reshape(1, -1), None, "nngp")
    di = jnp.apply_along_axis(lambda_kernel, axis=1, arr=training_inputs).flatten() + 1e-12
    if sample:
        indices = indices.at[0].set(sample_discrete(di))
    else:
        indices = indices.at[0].set(jnp.argmax(di))  # select first point, add to index 0
    
    if M == 1:
        indices = indices.astype(int)
        Z = training_inputs[indices]
        indices = perm[indices]
        return Z, indices
    ci = jnp.zeros((M - 1, N))  # [M,N]
    
    for m in range(M - 1):
        j = int(indices[m])  # int
        new_Z = training_inputs[j:j + 1]  # [1,D]
        dj = jnp.sqrt(di[j])  # float
        cj = ci[:m, j]  # [m, 1]
        Lraw = kernel_fn(training_inputs, new_Z, "nngp")
        L = jnp.round(jnp.squeeze(Lraw), 20)  # [N]
        L = L.at[j].add(1e-12) # L[j] += 1e-12  jitter
        ei = (L - jnp.dot(cj, ci[:m])) / dj
        ci = lax.dynamic_update_index_in_dim(ci, ei, m, 0) # ci[m, :] = ei
        try:
            di -= ei ** 2
        except FloatingPointError:
            pass
        di = jnp.clip(di, 0, None)
        if sample:
            indices = indices.at[m + 1].set(sample_discrete(di))
        else:
            indices = indices.at[m + 1].set(jnp.argmax(di)) # select first point, add to index 0
        # sum of di is tr(Kff-Qff), if this is small things are ok
        if jnp.sum(jnp.clip(di, 0, None)) < threshold:
            indices = indices[:m]
            logger.warning("ConditionalVariance: Terminating selection of inducing points early.")
            break
    
    indices = indices.astype(int)
    Z = training_inputs[indices]
    return Z

def sample_discrete(unnormalized_probs):
    unnormalized_probs = jnp.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = jnp.sum(unnormalized_probs)
    if normalization == 0:  # if all of the probabilities are numerically 0, sample uniformly
        logger.warning("Trying to sample discrete distribution with all 0 weights")
        return np.random.choice(a=N, size=1)[0]
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]