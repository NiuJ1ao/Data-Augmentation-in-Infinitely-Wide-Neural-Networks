import numpy as np
import jax.numpy as jnp
from jax import lax
from logger import get_logger
from tqdm import tqdm
from util import init_kernel_fn
logger = get_logger()

def random_select(X, M):
    if M > X.shape[0]:
        logger.warning(f"M is larger than the number of samples, M = {M}")
        return X
    indices = np.random.choice(X.shape[0], M, replace=False)
    return X[indices], jnp.array(indices)

def first_n(X, M):
    return X[:M, :], jnp.arange(M)

def greedy_variance(X, M, kernel_fn, sample=False, threshold=0.0):
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
    num_train = training_inputs.shape[0]
    input_shape = (1,) + training_inputs.shape[1:]
    def diag_kernel(x):
        return kernel_fn(x.reshape(input_shape), None, "nngp")
    di = jnp.apply_along_axis(diag_kernel, axis=1, arr=training_inputs.reshape((num_train, -1))).flatten() + 1e-12
    
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
    
    for m in tqdm(range(M - 1)):
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
    return Z, indices

def greedy_variance_generator(X, M, kernel_fn, sample=False, threshold=0.0):
    N = X.shape[0]
    perm = np.random.permutation(N)
    training_inputs = X[perm]
    indices = jnp.zeros(M, dtype=int) + N
    
    # diagnoal of kernel
    num_train = training_inputs.shape[0]
    input_shape = (1,) + training_inputs.shape[1:]
    def diag_kernel(x):
        return kernel_fn(x.reshape(input_shape), None, "nngp")
    di = jnp.apply_along_axis(diag_kernel, axis=1, arr=training_inputs.reshape((num_train, -1))).flatten() + 1e-12
    
    if sample:
        indices = indices.at[0].set(sample_discrete(di))
    else:
        indices = indices.at[0].set(jnp.argmax(di))  # select first point, add to index 0
    
    yield training_inputs[indices[0]], perm[indices.astype(int)[0]]
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
    
        index = indices.astype(int)[m + 1]
        Z = training_inputs[index]
        yield Z, index

def sample_discrete(unnormalized_probs):
    unnormalized_probs = jnp.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = jnp.sum(unnormalized_probs)
    if normalization == 0:  # if all of the probabilities are numerically 0, sample uniformly
        logger.warning("Trying to sample discrete distribution with all 0 weights")
        return np.random.choice(a=N, size=1)[0]
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]


class ConditionalVarianceGenerator():
    def __init__(self, X, M, kernel_fn, sample=False, threshold=0.0):
        self.sample = sample
        self.threshold = threshold
        self.X = X
        self.N = X.shape[0]
        self.M = M
        self.perm = np.random.permutation(self.N)
        self.training_inputs = X[self.perm]
        self.indices = jnp.zeros(M, dtype=int) + self.N
        self.kernel_fn = kernel_fn
        num_train = self.training_inputs.shape[0]
        input_shape = (1,) + self.training_inputs.shape[1:]
        def diag_kernel(x):
            return kernel_fn(x.reshape(input_shape), None, "nngp")
        self.di = jnp.apply_along_axis(diag_kernel, axis=1, arr=self.training_inputs.reshape((num_train, -1))).flatten() + 1e-12
        
        if sample:
            self.indices = self.indices.at[0].set(sample_discrete(self.di))
        else:
            self.indices = self.indices.at[0].set(jnp.argmax(self.di))  # select first point, add to index 0
        
    def __iter__(self):
        self.step = -1
        self.ci = jnp.zeros((self.M - 1, self.N))  # [M,N]
        return self
        
    def __next__(self):
        if self.step == -1:
            indices = self.indices.astype(int)
            Z = self.training_inputs[self.indices[0]]
            indices = self.perm[indices[0]]
            self.step += 1
            return Z, indices
        elif self.step < self.M - 1:
            j = int(self.indices[self.step])  # int
            new_Z = self.training_inputs[j:j + 1]  # [1,D]
            dj = jnp.sqrt(self.di[j])  # float
            cj = self.ci[:self.step, j]  # [m, 1]
            Lraw = self.kernel_fn(self.training_inputs, new_Z, "nngp")
            L = jnp.round(jnp.squeeze(Lraw), 20)  # [N]
            L = L.at[j].add(1e-12) # L[j] += 1e-12  jitter
            self.ei = (L - jnp.dot(cj, self.ci[:self.step])) / dj
            self.ci = lax.dynamic_update_index_in_dim(self.ci, self.ei, self.step, 0) # ci[m, :] = ei
            try:
                self.di -= self.ei ** 2
            except FloatingPointError:
                pass
            self.di = jnp.clip(self.di, 0, None)
            self.step += 1
            if self.sample:
                self.indices = self.indices.at[self.step].set(sample_discrete(self.di))
            else:
                self.indices = self.indices.at[self.step].set(jnp.argmax(self.di)) # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok
            if jnp.sum(jnp.clip(self.di, 0, None)) < self.threshold:
                self.indices = self.indices[:self.step]
                logger.warning("ConditionalVariance: Terminating selection of inducing points early.")
                raise StopIteration
            
            indices = self.indices.astype(int)
            Z = self.training_inputs[indices[self.step]]
            return Z, indices[self.step]
        else:
            raise StopIteration

def select_inducing_points(method, train, M, model=None, stds=None, model_params=None):
    if method == "random":
        inducing_points, indices = random_select(train, M)
    elif method == "first":
        inducing_points, indices = first_n(train, M)
    elif method == "greedy":
        kernel_fn = init_kernel_fn(model, stds, model_params)
        inducing_points, indices = greedy_variance(train, M, kernel_fn)
    logger.info(f"inducing_points shape: {inducing_points.shape}")
    return inducing_points, indices
