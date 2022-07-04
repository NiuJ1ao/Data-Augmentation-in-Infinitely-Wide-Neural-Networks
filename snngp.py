from collections import namedtuple
import jax.numpy as jnp
import numpy as np
from jax import scipy, value_and_grad, jit
from logger import get_logger
from scipy.optimize import minimize

logger = get_logger()

class _common_tensors():
    def __init__(self, A, B, LB, AAT, L):
        self.A = A
        self.B = B
        self.LB = LB
        self.AAT = AAT
        self.L = L

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
        
        if stds is None:
            self.stds = jnp.array([1., 1.], dtype=jnp.float64)
            logger.debug(f"{self.stds}, {self.stds.dtype}")
        else:
            self.stds = stds
            
        self.is_precomputed = False
            
        self.CommonTensors = namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L"])
    
    def _init_kernel_fn(self, params):
        model = self.model(W_std=params[0], b_std=params[1], **self.hyper_params)
        self.kernel_fn = model.kernel_fn
        return self.kernel_fn
    
    def _precomputation(self, params):
        x, _ = self.data
        M = self.num_inducing_points
        kernel_fn = self._init_kernel_fn(params)

        kuu = kernel_fn(self.inducing_points, None, self.get) # (M, M)
        kuf = kernel_fn(self.inducing_points, x, self.get) # (M, N)
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        
        self.is_precomputed = True
        self.cached_tensors = self.CommonTensors(A, B, LB, AAT, L)
        
        return self.cached_tensors
    
    def elbo(self):
        x, y = self.data
        N = x.shape[0]
        
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            AAT = self.cached_tensors.AAT
        else:
            cached_tensors = self._precomputation(self.stds)
            A = cached_tensors.A
            LB = cached_tensors.LB
            AAT = cached_tensors.AAT
        
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        
        bound = N * jnp.log(2 * jnp.pi) \
            + jnp.sum(jnp.log(jnp.diag(LB))) \
            + N * jnp.log(self.sigma_squared) \
            + y @ y.T / self.sigma_squared \
            - c.T @ c

        regulariser = jnp.trace(self.kernel_fn(x, None, self.get)) / self.sigma_squared - jnp.trace(AAT)
        
        elbo  = -0.5 * (bound + regulariser)
        
        return elbo
            
    def _neg_elbo(self, params):
        x, y = self.data
        N = x.shape[0]
        M = self.num_inducing_points
        kernel_fn = self._init_kernel_fn(params)
        
        kuu = kernel_fn(self.inducing_points, None, self.get) # (M, M)
        kuf = kernel_fn(self.inducing_points, x, self.get) # (M, N)
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        
        bound = N * jnp.log(2 * jnp.pi) \
            + jnp.sum(jnp.log(jnp.diag(LB))) \
            + N * jnp.log(self.sigma_squared) \
            + y @ y.T / self.sigma_squared \
            - c.T @ c

        regulariser = jnp.trace(kernel_fn(x, None, self.get)) / self.sigma_squared - jnp.trace(AAT)
        
        neg_elbo = 0.5 * (bound + regulariser)
        
        return neg_elbo
    
    def upper_bound(self):
        x, y = self.data
        N = x.shape[0]
        M = self.num_inducing_points
        # kernel_fn = self._init_kernel_fn(params)
        
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            AAT = self.cached_tensors.AAT
        else:
            cached_tensors = self._precomputation(self.stds)
            A = cached_tensors.A
            LB = cached_tensors.LB
            AAT = cached_tensors.AAT
        
        # kuu = kernel_fn(self.inducing_points, None, self.get) # (M, M)
        # kuf = kernel_fn(self.inducing_points, x, self.get) # (M, N)
        
        # L = jnp.linalg.cholesky(kuu) # (M, M)
        
        # # normaliser
        # A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        # AAT = A @ A.T # (M, M)
        # B = jnp.eye(M) + AAT # (M, M)
        # LB = jnp.linalg.cholesky(B) # (M, M)
        
        log_terms = N * jnp.log(2 * jnp.pi) \
            + jnp.sum(jnp.log(jnp.diag(LB))) \
            + N * jnp.log(self.sigma_squared)
                
        # inverse (woodbury identity)
        trace_term = jnp.trace(self.kernel_fn(x, None, self.get)) - jnp.trace(AAT) * self.sigma_squared
        alpha = trace_term + self.sigma_squared
        alpha_sqrt = jnp.sqrt(alpha)
        
        A = A * self.sigma / alpha_sqrt # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / alpha_sqrt # (M, )
        
        quad_term = y @ y.T / alpha - c.T @ c
        
        upper_bound = -0.5 * (log_terms + quad_term)
        
        return upper_bound
    
    def evaluate(self):
        """
        """
        return self.upper_bound() - self.elbo()
        
    def optimize(self):
        logger.info("Optimizing...")
        
        grad_elbo = jit(value_and_grad(self._neg_elbo))
        
        def grad_elbo_wrapper(params):
            value, grads = grad_elbo(params)
            # scipy.optimize.minimize cannot handle JAX DeviceArray
            value, grads = np.array(value, dtype=np.float64), np.array(grads, dtype=np.float64)
            return value, grads
        
        res = minimize(fun=grad_elbo_wrapper, x0=self.stds, method="L-BFGS-B", jac=True)
        logger.debug(f"{res}")
        logger.info(f"Optimized for {res.nit} iters; Success: {res.success}; Result: {res.x}")
        
        if res.success == True:
            self.stds = jnp.asarray(res.x)
            self._precomputation(self.stds)
        
        return self.stds
                
    def predict(self, test, opt_params):
        """Posterior distribution
        """
        x, y = self.data
        M = self.num_inducing_points
        kernel_fn = self._init_kernel_fn(opt_params)

        kuu = kernel_fn(self.inducing_points, None, self.get) # (M, M)
        kuf = kernel_fn(self.inducing_points, x, self.get) # (M, N)
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        
        L_inv = jnp.linalg.inv(L) # (M, M)
        LB_inv = jnp.linalg.inv(LB) # (M, M)
        B_inv = jnp.linalg.inv(B) # (M, M)
        
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        
        kss = kernel_fn(test, test, self.get)
        ksu = kernel_fn(test, self.inducing_points, self.get)
        kus = ksu.T
        
        mean = ksu @ L_inv.T @ LB_inv.T @ c
        cov = kss - ksu @ L_inv.T @ (jnp.eye(M) - B_inv) @ LB_inv.T @ kus

        return mean, cov
        
