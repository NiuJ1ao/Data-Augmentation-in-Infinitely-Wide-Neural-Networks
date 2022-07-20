from collections import namedtuple
import numpy as np
import jax.numpy as jnp
from jax import scipy, value_and_grad
from logger import get_logger
from scipy.optimize import minimize
from util import softplus, softplus_inv, batch_kernel
from snngp import kernel_diagonal, init_kernel_fn
from snngp import SNNGP
from jax.config import config
config.update("jax_enable_x64", True)

logger = get_logger()

class SummationKernel():
    def __init__(self, kernel_fn,
                 orbit_size,
                 augmenter,
                 batch_size=0,
                 **augment_params):
        self.kernel_fn = kernel_fn
        self.orbit_size = orbit_size
        self.augmenter = augmenter
        self.augment_params = augment_params
        self.batch_size = batch_size
        self.get = "nngp"
    
    def generate_orbit(self, x):
        return jnp.expand_dims(x, 0)
    
    def single_sum_kernel(self, x1, x2):
        N = x1.shape[0]
        M = x2.shape[0]
        input_shape = (1,) + x1.shape[1:]
        x1 = x1.reshape((N, -1))
        
        def wrapper(x):
            x = x.reshape(input_shape)
            orbit = self.generate_orbit(x)
            return jnp.sum(self.kernel_fn(orbit, x2, self.get), axis=0) # sum(OxM) -> (M,)
        kernel = jnp.apply_along_axis(wrapper, 1, x1)
        return kernel # NxM
    
    def double_sum_kernel(self, x1, x2):
        N = x1.shape[0]
        M = x2.shape[0]
        kernel = jnp.zeros((N, M))
        for i in x1:
            for j in x2:
                orbit_i = self.generate_orbit(i)
                orbit_j = self.generate_orbit(j)
                kernel_ij = jnp.sum(self.kernel_fn(orbit_i, orbit_j, self.get)) # sum(OxO) -> ()
                kernel = kernel.at[i, j].set(kernel_ij)
        return kernel
    
    def double_sum_kernel_diag(self, data):
        N = data.shape[0]
        input_shape = (1,) + data.shape[1:]
        data = data.reshape((N, -1))
        
        def wrapper(x):
            x = x.reshape(input_shape)
            orbit = self.generate_orbit(x)
            return jnp.sum(self.kernel_fn(orbit, None, self.get))
        diag = jnp.apply_along_axis(wrapper, 1, data).flatten()
        return diag
    
    def __call__(self, x1, x2, diff_domain=(False, False), diag=False):
        if diag:
            assert jnp.allclose(x1, x2)
            if diff_domain == (True, True):
                return kernel_diagonal(self.kernel_fn, x1)
            return self.double_sum_kernel_diag(x1)
        else:
            if diff_domain == (True, True):
                return batch_kernel(self.kernel_fn, self.batch_size, x1, x2)
            if diff_domain[0]:
                return self.single_sum_kernel(x1, x2)
            if diff_domain[1]:
                return self.single_sum_kernel(x2, x1).T
            return self.double_sum_kernel(x1, x2)

class iSNNGP(SNNGP):
    def __init__(self, 
                 model, 
                 model_params, 
                 orbit_size,
                 train_data, 
                 inducing_points,
                 augmenter,
                 augment_params,
                 num_latent_gps, 
                 noise_variance=1, 
                 init_stds=None,
                 batch_size=0):
        super().__init__(
            model, 
            model_params, 
            train_data, 
            inducing_points, 
            num_latent_gps, 
            noise_variance, 
            init_stds,
            batch_size
        )
        self.orbit_size = orbit_size
        self.augmenter = augmenter
        self.augment_params = augment_params
    
    def _init_kernel_fn(self, stds):
        kernel_fn = init_kernel_fn(self.model, stds, self.hyper_params)
        self.kernel_fn = SummationKernel(kernel_fn, self.orbit_size, self.augmenter, self.augment_params)
    
    def _precomputation(self):
        x, _ = self.data
        M = self.num_inducing_points
        logger.debug(f"stds: {self.stds}")
        self._init_kernel_fn(self.stds)

        kuu = self.kernel_fn(self.inducing_points, self.inducing_points, diff_domain=(True, True)) \
            + self.jitter * jnp.eye(M) # (M, M)
        logger.debug(f"kuu: {kuu.shape}")
        kuf = self.kernel_fn(self.inducing_points, x, diff_domain=(True, False)) # (M, N)
        logger.debug(f"kuf: {kuf.shape}")
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        # logger.debug(f"L: {L.shape}")
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        # logger.debug(f"A: {A.shape}")
        AAT = A @ A.T # (M, M)
        # logger.debug(f"AAT: {AAT.shape}")
        B = jnp.eye(M) + AAT # (M, M)
        # logger.debug(f"B: {B.shape}")
        LB = jnp.linalg.cholesky(B) # (M, M)
        # logger.debug(f"LB: {LB.shape}")
        
        self.is_precomputed = True
        self.cached_tensors = self.CommonTensors(A, B, LB, AAT, L)
        
        return self.cached_tensors
    
    def upper_bound(self):
        x, y = self.data
        M = self.num_inducing_points
        
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            AAT = self.cached_tensors.AAT
        else:
            cached_tensors = self._precomputation()
            A = cached_tensors.A
            LB = cached_tensors.LB
            AAT = cached_tensors.AAT
        
        const = self._const_term()
        logdet = self._logdet_term(LB)
        
        trace_k = jnp.sum(self.kernel_fn(x, x, diff_domain=(False, False), diag=True))
        trace_q = jnp.trace(AAT) * self.sigma_squared
        trace = trace_k - trace_q
        
        alpha = trace + self.sigma_squared
        alpha_sqrt = jnp.sqrt(alpha)
        
        A = A * self.sigma / alpha_sqrt # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / alpha_sqrt # (M, D)
        
        quad_term = -0.5 * (jnp.sum(jnp.square(y)) / alpha - jnp.sum(jnp.square(c)))
        
        upper_bound = const + logdet + quad_term
        return upper_bound
    
    def log_likelihood(self):
        x, y = self.data
        
        self._init_kernel_fn(self.stds)
        cov = self.kernel_fn(x, x)
        N = cov.shape[0]
        
        chol = scipy.linalg.cholesky(
                cov + self.sigma * jnp.eye(N), lower=True)
        logger.debug(f"{chol.shape}, {y.shape}")
        alpha = scipy.linalg.cho_solve((chol, True), y)
        logger.debug(alpha.shape)

        ll = jnp.sum(-0.5 * jnp.einsum('ik,ik->k', y, alpha) -
                    jnp.sum(jnp.log(jnp.diag(chol))) - (N / 2.) * jnp.log(2. * jnp.pi))
        logger.debug(ll)
        
        return ll
      
    def _training_loss(self, params):
        x, _ = self.data
        M = self.num_inducing_points
        
        # stds, inducing_points = self.unpack_params(params)
        stds = softplus(params)
        self._init_kernel_fn(stds)
        
        # ELBO
        kuu = self.kernel_fn(self.inducing_points, self.inducing_points, diff_domain=(True, True)) \
            + self.jitter * jnp.eye(M) # (M, M)
        logger.debug(f"kuu: {kuu.shape}")
        kuf = self.kernel_fn(self.inducing_points, x, diff_domain=(True, False)) # (M, N)
        logger.debug(f"kuf: {kuf.shape}")

        L = jnp.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        
        logdet = self._logdet_term(LB)
        trace = self._trace_term(AAT)
        quad = self._quad_term(A, LB)
        const = self._const_term()
        loss = - const - logdet - quad - trace
        return loss
        
    def predict(self, test, diag=False):
        _, y = self.data
        
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            L = self.cached_tensors.L
        else:
            cached_tensors = self._precomputation()
            A = cached_tensors.A
            LB = cached_tensors.LB
            L = cached_tensors.L
        
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        kus = self.kernel_fn(self.inducing_points, test, diff_domain=(True, False))
        
        tmp1 = scipy.linalg.solve_triangular(L, kus, lower=True)
        tmp2 = scipy.linalg.solve_triangular(LB, tmp1, lower=True)
        
        mean = tmp2.T @ c
        if diag:
            cov = self.kernel_fn(test, test, diff_domain=(False, False), diag=True) + jnp.sum(jnp.square(tmp2), 0) - jnp.sum(jnp.square(tmp1), 0)
            cov = jnp.tile(cov[:, None], [1, self.num_latent_gps]) # (N, 1) -> (N, P)
        else:
            cov = self.kernel_fn(test, test, diff_domain=(False, False), diag=False) + tmp2.T @ tmp2 - tmp1.T @ tmp1
            cov = jnp.tile(cov[None, ...], [self.num_latent_gps, 1, 1]) # (1, N, N) -> (P, N, N)

        return mean, cov


if __name__ == "__main__":
    # x1 = jnp.arange(15).reshape((3,5))
    # x2 = jnp.arange(10).reshape((2,5))
    x1 = jnp.arange(75).reshape((3,5,5))
    x2 = jnp.arange(50).reshape((2,5,5))
    print(x1.at[0][0][0].set(1))