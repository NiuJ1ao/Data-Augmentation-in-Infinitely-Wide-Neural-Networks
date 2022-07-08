from collections import namedtuple
import jax.numpy as jnp
import numpy as np
from jax import scipy, value_and_grad, jit
from logger import get_logger
from scipy.optimize import minimize
from util import softplus, softplus_inv
from jax.config import config
config.update("jax_enable_x64", True)

logger = get_logger()

class SNNGP():
    def __init__(self, 
                 model, 
                 hyper_params, 
                 train_data, 
                 inducing_points, 
                 num_latent_gps, 
                 noise_variance=1, 
                 stds=None):
        
        self.data = train_data
        self.model = model
        self.inducing_points = inducing_points
        self.num_inducing_points = inducing_points.shape[0]
        self.num_latent_gps = num_latent_gps
        self.sigma = jnp.sqrt(noise_variance)
        self.sigma_squared = noise_variance
        self.hyper_params = hyper_params
        self.get = "nngp"
        self.jitter = 1e-6
        if stds is None:
            self.stds = jnp.array([1., 1.], dtype=jnp.float64)
        else:
            self.stds = stds
            
        self.is_precomputed = False
            
        self.CommonTensors = namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L", "trace_x"])
    
    def _init_kernel_fn(self, stds):
        model = self.model(W_std=stds[0], b_std=stds[1], **self.hyper_params)
        self.kernel_fn = model.kernel_fn
        return self.kernel_fn
        
    # def pack_params(self):
    #     return jnp.concatenate([softplus_inv(self.stds), self.inducing_points.ravel()])
    
    # def unpack_params(self, params):
    #     stds = softplus(params[:2])
    #     inducing_points = params[2:].reshape((self.num_inducing_points, -1))
    #     return stds, inducing_points
    
    def _precomputation(self):
        x, _ = self.data
        M = self.num_inducing_points
        kernel_fn = self._init_kernel_fn(self.stds)

        kuu = kernel_fn(self.inducing_points, None, self.get) + self.jitter * jnp.eye(M) # (M, M)        
        logger.debug(f"kuu: {kuu.shape}")
        kuf = kernel_fn(self.inducing_points, x, self.get) # (M, N)
        logger.debug(f"kuf: {kuf.shape}")
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        logger.debug(f"L: {L.shape}")
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        logger.debug(f"A: {A.shape}")
        AAT = A @ A.T # (M, M)
        logger.debug(f"AAT: {AAT.shape}")
        B = jnp.eye(M) + AAT # (M, M)
        logger.debug(f"B: {B.shape}")
        LB = jnp.linalg.cholesky(B) # (M, M)
        logger.debug(f"LB: {LB.shape}")
        
        trace_x = jnp.trace(self.kernel_fn(x, None, self.get))
        
        self.is_precomputed = True
        self.cached_tensors = self.CommonTensors(A, B, LB, AAT, L, trace_x)
        
        return self.cached_tensors
    
    def _logdet_term(self, LB):
        """log determinant term
        """
        x, y = self.data
        N = x.shape[0]
        out_dim = y.shape[1]

        half_logdet_B = jnp.sum(jnp.log(jnp.diag(LB)))
        log_sigma_sq = N * jnp.log(self.sigma_squared)
        
        logdet = -out_dim * (half_logdet_B + 0.5 * (log_sigma_sq))
        # logger.debug(f"logdet: {logdet}")
        return logdet
    
    def _trace_term(self, trace_x, AAT):
        """trace term
        """
        x, y = self.data
        out_dim = y.shape[1]
        
        trace_k = trace_x / self.sigma_squared
        trace_q = jnp.trace(AAT)
        trace = trace_k - trace_q
        # logger.debug(f"trace: {trace}")
        return -0.5 * out_dim * trace
    
    def _quad_term(self, A, LB):
        _, y = self.data
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, D)
        quad = -0.5 * (jnp.sum(y.T @ y) / self.sigma_squared - jnp.sum(c.T @ c))
        # logger.debug(f"quad: {quad}")
        return quad
    
    def _const_term(self):
        x, y = self.data
        N = x.shape[0]
        out_dim = y.shape[1]
        return -0.5 * N * out_dim * jnp.log(2 * jnp.pi)
    
    def lower_bound(self):
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            AAT = self.cached_tensors.AAT
            trace_x = self.cached_tensors.trace_x
        else:
            cached_tensors = self._precomputation()
            A = cached_tensors.A
            LB = cached_tensors.LB
            AAT = cached_tensors.AAT
            trace_x = cached_tensors.trace_x
        
        logdet = self._logdet_term(LB)
        trace = self._trace_term(trace_x, AAT)
        quad = self._quad_term(A, LB)    
        const = self._const_term()
        elbo = const + logdet + quad + trace
        return elbo
            
    def _neg_elbo(self, params):
        x, _ = self.data
        M = self.num_inducing_points
        
        # stds, inducing_points = self.unpack_params(params)
        stds = softplus(params)
        inducing_points = self.inducing_points
        kernel_fn = self._init_kernel_fn(stds)
        
        kuu = kernel_fn(inducing_points, None, self.get) + self.jitter * jnp.eye(M) # (M, M)
        kuf = kernel_fn(inducing_points, x, self.get) # (M, N)
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        
        logdet = self._logdet_term(LB)
        trace_x = jnp.trace(self.kernel_fn(x, None, self.get)) # TODO: accelerate trace_x
        trace = self._trace_term(trace_x, AAT)
        quad = self._quad_term(A, LB)    
        const = self._const_term()
        elbo = const + logdet + quad + trace
        return -elbo
    
    def upper_bound(self):
        x, y = self.data
        N = x.shape[0]
        M = self.num_inducing_points
        
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            AAT = self.cached_tensors.AAT
            trace_x = self.cached_tensors.trace_x
        else:
            cached_tensors = self._precomputation()
            A = cached_tensors.A
            LB = cached_tensors.LB
            AAT = cached_tensors.AAT
            trace_x = cached_tensors.trace_x
        
        const = self._const_term()
        logdet = self._logdet_term(LB)
        
        trace_k = trace_x
        trace_q = jnp.trace(AAT) * self.sigma_squared
        trace = trace_k - trace_q
        
        alpha = trace + self.sigma_squared
        alpha_sqrt = jnp.sqrt(alpha)
        
        A = A * self.sigma / alpha_sqrt # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / alpha_sqrt # (M, D)
        
        quad_term = -0.5 * (jnp.sum(y.T @ y) / alpha - jnp.sum(c.T @ c))
        
        upper_bound = const + logdet + quad_term
        return upper_bound
    
    def evaluate(self):
        """
        """
        return self.upper_bound() - self.lower_bound()
        
    def optimize(self):
        logger.info("Optimizing...")

        grad_elbo = jit(value_and_grad(self._neg_elbo))
        
        def grad_elbo_wrapper(params):
            value, grads = grad_elbo(params)
            # scipy.optimize.minimize cannot handle JAX DeviceArray
            value, grads = np.array(value, dtype=np.float64), np.array(grads, dtype=np.float64)
            return value, grads
        
        # res = minimize(fun=grad_elbo_wrapper, x0=self.pack_params(), method="L-BFGS-B", jac=True, options={"disp": True})
        res = minimize(fun=grad_elbo_wrapper, x0=softplus_inv(self.stds), method="L-BFGS-B", jac=True)
        logger.debug(f"{res}")
        logger.info(f"Optimized for {res.nit} iters; Success: {res.success}; Result: {res.x}")
        
        # if res.success == True:
        # stds, inducing_points = self.unpack_params(jnp.asarray(res.x))
        # self.inducing_points = inducing_points
        stds = softplus(jnp.asarray(res.x))
        self.stds = stds
        self._precomputation()
    
        return self.stds
        
    def predict(self, test, diag=False):
        _, y = self.data
        
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            L = self.cached_tensors.L
        else:
            cached_tensors = self._precomputation(self.stds)
            A = cached_tensors.A
            LB = cached_tensors.LB
            L = cached_tensors.L
        
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        
        kss = self.kernel_fn(test, test, self.get)
        kus = self.kernel_fn(self.inducing_points, test, self.get)
        
        tmp1 = scipy.linalg.solve_triangular(L, kus, lower=True)
        tmp2 = scipy.linalg.solve_triangular(LB, tmp1, lower=True)
        
        mean = tmp2.T @ c
        cov = kss + tmp2.T @ tmp2 - tmp1.T @ tmp1
        if diag:
            cov = jnp.diag(cov)
            cov = jnp.tile(cov[:, None], [1, self.num_latent_gps]) # (N, 1) -> (N, P)
        else:
            cov = jnp.tile(cov[None, ...], [self.num_latent_gps, 1, 1]) # (1, N, N) -> (P, N, N)

        return mean, cov
