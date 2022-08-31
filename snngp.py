from collections import namedtuple
import numpy as np
import jax.numpy as jnp
from jax import scipy, value_and_grad, jit
from logger import get_logger
from scipy.optimize import minimize
from util import softplus, softplus_inv, batch_kernel, init_kernel_fn, kernel_diagonal
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
                 init_stds=None,
                 batch_size=0):
        
        self.data = train_data
        self.num_train = train_data[1].shape[0]
        if len(train_data[1].shape) == 1:
            self.out_dim = 1
        else:
            self.out_dim = train_data[1].shape[1]
        self.model = model
        self.inducing_points = inducing_points
        self.num_inducing_points = inducing_points.shape[0]
        self.num_latent_gps = num_latent_gps
        self.sigma = jnp.sqrt(noise_variance)
        self.sigma_squared = noise_variance
        self.hyper_params = hyper_params
        self.batch_size = batch_size
        self.get = "nngp"
        self.jitter = 1e-6
        if init_stds is None:
            self.stds = jnp.array([1., 1.], dtype=jnp.float64)
        else:
            self.stds = init_stds
            
        self.is_precomputed = False
            
        self.CommonTensors = namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L"])
    
    # def pack_params(self):
    #     return jnp.concatenate([softplus_inv(self.stds), self.inducing_points.ravel()])
    
    # def unpack_params(self, params):
    #     stds = softplus(params[:2])
    #     inducing_points = params[2:].reshape((self.num_inducing_points, -1))
    #     return stds, inducing_points
    
    def _precomputation(self):
        x, _ = self.data
        M = self.num_inducing_points
        logger.debug(f"stds: {self.stds}")
        self.kernel_fn = init_kernel_fn(self.model, self.stds, self.hyper_params)

        kuu = batch_kernel(self.kernel_fn, self.batch_size, self.inducing_points, self.inducing_points)
        kuu += self.jitter * jnp.eye(M) # (M, M)
        logger.debug(f"kuu: {kuu.shape}")
        kuf = batch_kernel(self.kernel_fn, self.batch_size, x, self.inducing_points).T
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
    
    def _logdet_term(self, LB):
        """log determinant term
        """        
        half_logdet_B = jnp.sum(jnp.log(jnp.diag(LB)))
        log_sigma_sq = self.num_train * jnp.log(self.sigma_squared)
        
        logdet = -self.out_dim * (half_logdet_B + 0.5 * (log_sigma_sq))
        # logger.debug(f"logdet: {logdet}")
        return logdet
    
    def _trace_term(self, AAT):
        """trace term
        """
        x, _ = self.data
        
        trace_k = jnp.sum(kernel_diagonal(self.kernel_fn, x)) / self.sigma_squared
        trace_q = jnp.trace(AAT)
        trace = trace_k - trace_q
        # logger.debug(f"trace: {trace}")
        return -0.5 * self.out_dim * trace
    
    def _quad_term(self, A, LB):
        _, y = self.data
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, D)
        quad = -0.5 * (jnp.sum(jnp.square(y)) / self.sigma_squared - jnp.sum(jnp.square(c)))
        # logger.debug(f"quad: {quad}")
        return quad
    
    def _const_term(self):
        return -0.5 * self.num_train * self.out_dim * jnp.log(2 * jnp.pi)
    
    def lower_bound(self):
        if self.is_precomputed:
            A = self.cached_tensors.A
            LB = self.cached_tensors.LB
            AAT = self.cached_tensors.AAT
        else:
            cached_tensors = self._precomputation()
            A = cached_tensors.A
            LB = cached_tensors.LB
            AAT = cached_tensors.AAT
        
        logdet = self._logdet_term(LB)
        trace = self._trace_term(AAT)
        logger.debug(f"trace term: {trace}")
        quad = self._quad_term(A, LB)    
        const = self._const_term()
        elbo = const + logdet + quad + trace
        return elbo
    
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
        
        trace_k = jnp.sum(kernel_diagonal(self.kernel_fn, x))
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
    
    def log_marginal_likelihood(self):
        x, y = self.data
        
        self.kernel_fn = init_kernel_fn(self.model, self.stds, self.hyper_params)
        cov = self.kernel_fn(x, x, get=self.get)
        N = cov.shape[0]
            
        chol = scipy.linalg.cholesky(
                cov + self.sigma_squared * jnp.eye(N), lower=True)
        logger.debug(f"{chol.shape}, {y.shape}")
        alpha = scipy.linalg.cho_solve((chol, True), y)
        logger.debug(alpha.shape)

        ll = jnp.sum(-0.5 * jnp.einsum('ik,ik->k', y, alpha) -
                    jnp.sum(jnp.log(jnp.diag(chol))) - (N / 2.) * jnp.log(2. * jnp.pi))
        logger.debug(ll)
        
        return ll
      
    def pack_params(self):
        return jnp.append(softplus_inv(self.stds), softplus_inv(self.sigma))
    
    def unpack_params(self, params):
        self.stds = softplus(params[:2])
        self.sigma = softplus(params[-1])
        self.sigma_squared = jnp.square(self.sigma)
        return self.stds, self.sigma_squared

    def _training_loss(self, params):
        x, _ = self.data
        M = self.num_inducing_points
        
        self.unpack_params(params)
        # stds = softplus(params)
        self.kernel_fn = init_kernel_fn(self.model, self.stds, self.hyper_params)
        
        # ELBO
        kuu = batch_kernel(self.kernel_fn, self.batch_size, self.inducing_points, self.inducing_points) + self.jitter * jnp.eye(M) # (M, M)
        kuf = batch_kernel(self.kernel_fn, self.batch_size, x, self.inducing_points).T

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
            
    # def optimize(self):
    #     logger.info("Optimizing...")

    #     grad_elbo = jit(value_and_grad(self._training_loss))
        
    #     def grad_elbo_wrapper(params):
    #         value, grads = grad_elbo(params)
    #         # scipy.optimize.minimize cannot handle JAX DeviceArray
    #         value, grads = np.array(value, dtype=np.float64), np.array(grads, dtype=np.float64)
    #         return value, grads
        
    #     res = minimize(fun=grad_elbo_wrapper, x0=self.pack_params(), method="L-BFGS-B", jac=True, options={"maxiter": 5000, "disp": True})
    #     # res = optimize.minimize(fun=self._training_loss, x0=self.pack_params(), method="BFGS", options={"maxiter": 5000})
    #     # logger.debug(f"{res}")
    #     logger.info(f"Optimized for {res.nit} iters; Success: {res.success}; Result: {res.x}")
        
    #     # if res.success == True:
    #     stds, inducing_points = self.unpack_params(jnp.asarray(res.x))
    #     self.inducing_points = inducing_points
    #     self.stds = stds
    #     self.is_precomputed = False
    
    #     return self.stds
    
    def optimize(self, compile=False, disp=True):
        grad_elbo = value_and_grad(self._training_loss)
        if compile:
            grad_elbo = jit(grad_elbo)

        def grad_elbo_wrapper(params):
            value, grads = grad_elbo(params)
            # scipy.optimize.minimize cannot handle JAX DeviceArray
            value, grads = np.array(value, dtype=np.float64), np.array(grads, dtype=np.float64)
            return value, grads
        
        logger.info("Optimizing...")
        # res = minimize(fun=grad_elbo_wrapper, x0=softplus_inv(self.stds), method="L-BFGS-B", jac=True, options={"maxiter": 5000, "disp": disp})
        res = minimize(fun=grad_elbo_wrapper, x0=self.pack_params(), method="L-BFGS-B", jac=True, options={"maxiter": 5000, "disp": disp})
        # logger.debug(f"{res}")
        
        # self.stds = softplus(jnp.asarray(res.x))
        self.unpack_params(jnp.asarray(res.x))
        self.is_precomputed = False
        logger.info(f"Optimized for {res.nit} iters; Success: {res.success}; Result: {self.stds}, {self.sigma_squared}")
    
        return res.success, self.stds, self.sigma_squared
        
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
        kus = batch_kernel(self.kernel_fn, self.batch_size, test, self.inducing_points).T
        
        tmp1 = scipy.linalg.solve_triangular(L, kus, lower=True)
        tmp2 = scipy.linalg.solve_triangular(LB, tmp1, lower=True)
        
        mean = tmp2.T @ c
        if diag:
            cov = kernel_diagonal(self.kernel_fn, test) + jnp.sum(jnp.square(tmp2), 0) - jnp.sum(jnp.square(tmp1), 0)
            cov = jnp.tile(cov[:, None], [1, self.num_latent_gps]) # (N, 1) -> (N, P)
        else:
            cov = self.kernel_fn(test, test, self.get) + tmp2.T @ tmp2 - tmp1.T @ tmp1
            cov = jnp.tile(cov[None, ...], [self.num_latent_gps, 1, 1]) # (1, N, N) -> (P, N, N)

        return mean, cov
