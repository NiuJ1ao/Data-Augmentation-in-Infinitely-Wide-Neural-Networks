from scipy.optimize import minimize
from neural_tangents import empirical_nngp_fn
import jax.numpy as np
from jax import scipy, value_and_grad, jit
from logger import get_logger
logger = get_logger()

class SNNGP():
    def __init__(self, model, data, inducing_points, num_latent_gps, noise_variance=1):
        self.model = model
        # self.kernel_fn = model.kernel_fn 
        self.kernel_fn = empirical_nngp_fn(model.apply_fn)
        # diag_kernel_fn = empirical_nngp_fn(model.apply_fn, diagonal_axes=)
        self.data = data
        self.inducing_points = inducing_points
        self.num_inducing_points = inducing_points.shape[0]
        self.num_latent_gps = num_latent_gps
        self.sigma = np.sqrt(noise_variance)
        self.sigma_squred = noise_variance
    
    def log_marginal_likelihood_bound(self, params):
        x, y = self.data
        N = x.shape[0]
        M = self.num_inducing_points
        # sigma = y - np.zeros_like(y)

        kuu = self.kernel_fn(self.inducing_points, None,  params) # (M, M)
        kuf = self.kernel_fn(self.inducing_points, x,  params) # (M, N)
        
        L = np.linalg.cholesky(kuu) # (M, M)
        A = scipy.linalg.solve_triangular(L, kuf, lower=True) / self.sigma # (M, N)
        AAT = A @ A.T # (M, M)
        B = np.eye(M) + AAT # (M, M)
        LB = np.linalg.cholesky(B) # (M, M)
        
        c = scipy.linalg.solve_triangular(LB, A @ y, lower=True) / self.sigma # (M, )
        
        bound = N * np.log(2 * np.pi) \
            + np.sum(np.log(np.diag(LB))) \
            + N * np.log(self.sigma_squred) \
            + y @ y.T / self.sigma_squred \
            - c.T @ c
        regulariser = np.trace(self.kernel_fn(x, None, params)) - np.trace(AAT)
        elbo  = -0.5 * (bound + regulariser)
        logger.debug(f"elbo: {elbo}")
        
        return elbo
    
    def optimize(self):
        logger.debug(f"optimizing...")
        
        grad = jit(value_and_grad(lambda params: -self.log_marginal_likelihood_bound(params)))
        
        def objective(params):
            value, grads = grad(params)
            # scipy.optimize.minimize cannot handle
            # JAX DeviceArray directly. a conversion
            # to Numpy ndarray is needed.
            return np.array(value), np.array(grads)
        
        params = self.model.params
        logger.debug(type(params))
        logger.debug(f"{len(params)}")
        packed_params = self.model.pack_params()
        logger.debug(self.model.unpack_params(packed_params))
        assert False
        
        res = minimize(objective, self.model.params, method='L-BFGS-B', jac=True, options={'disp': True})
        logger.debug(f"optimization finished in {res.nit} iterations")
        
        self.model.update_params(res.x)
        new_elbo = self.log_marginal_likelihood_bound(self.model.params)
        logger.debug(f"new_elbo: {new_elbo}")
        return res.x
        
        
        
