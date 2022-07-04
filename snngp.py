import jax.numpy as jnp
import numpy as np
from jax import scipy, grad, value_and_grad, jit, random
from logger import get_logger
from optimizer import lbfgs
from scipy.optimize import minimize
from tqdm import tqdm

from util import split_key
logger = get_logger()

class SNNGP():
    def __init__(self, model, hyper_params, train_data, inducing_points, num_latent_gps, noise_variance=1, stds=None):
        x, y = train_data
        self.model = model
        self.data = (x, y)
        self.inducing_points = inducing_points
        self.num_inducing_points = inducing_points.shape[0]
        self.num_latent_gps = num_latent_gps
        self.sigma = jnp.sqrt(noise_variance)
        self.sigma_squred = noise_variance
        self.hyper_params = hyper_params
        
        if stds is None:
            _, key = split_key()
            self.stds = jnp.array([1., 1.], dtype=jnp.float64)
            logger.debug(f"{self.stds}, {self.stds.dtype}")
        else:
            self.stds = stds
    
    def neg_log_marginal_likelihood_bound(self, params):
        x, y = self.data
        N = x.shape[0]
        M = self.num_inducing_points
        model = self.model(W_std=params[0], b_std=params[1], **self.hyper_params)
        kernel_fn = model.kernel_fn
        get = "nngp"
        
        kuu = kernel_fn(self.inducing_points, None, get) # (M, M)
        kuf = kernel_fn(self.inducing_points, x, get) # (M, N)
        
        L = jnp.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = jnp.eye(M) + AAT # (M, M)
        LB = jnp.linalg.cholesky(B) # (M, M)
        
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        
        bound = N * jnp.log(2 * jnp.pi) \
            + jnp.sum(jnp.log(jnp.diag(LB))) \
            + N * jnp.log(self.sigma_squred) \
            + y @ y.T / self.sigma_squred \
            - c.T @ c
        regulariser = jnp.trace(kernel_fn(x, None, get)) - jnp.trace(AAT)
        elbo  = -0.5 * (bound + regulariser)
        # logger.debug(f"elbo: {elbo}")
        
        return -elbo
    
    def evaluate(self):
        """ KL divergence
        """
        
    
    # def optimize(self):
    #     grad_elbo = jit(grad(self.neg_log_marginal_likelihood_bound))
        
    #     opt_init, opt_update, get_params = lbfgs(max_iter=100)
    #     opt_state = opt_init(self.stds)
    #     for i in tqdm(range(10)):
    #         opt_state = opt_update(i, grad_elbo(get_params(opt_state)), opt_state)
    #     logger.debug(f"{get_params(opt_state)}")
        
    def optimize(self):
        logger.info("Optimizing...")
        
        grad_elbo = jit(value_and_grad(self.neg_log_marginal_likelihood_bound))
        
        def grad_elbo_wrapper(params):
            value, grads = grad_elbo(params)
            # scipy.optimize.minimize cannot handle JAX DeviceArray
            value, grads = np.array(value, dtype=np.float64), np.array(grads, dtype=np.float64)
            return value, grads
        
        res = minimize(fun=grad_elbo_wrapper, x0=self.stds, method="L-BFGS-B", jac=True)
        logger.debug(f"{res}")
        logger.info(f"Optimized for {res.nit} iters; Success: {res.success}; Result: {res.x}")
        
        return res.x
        
