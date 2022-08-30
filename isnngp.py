import jax.numpy as jnp
from jax import scipy
from tqdm import tqdm
from logger import get_logger
from util import softplus, batch_kernel
from snngp import kernel_diagonal, init_kernel_fn
from snngp import SNNGP
from jax.config import config
config.update("jax_enable_x64", True)

logger = get_logger()

class SummationKernel():
    def __init__(self, kernel_fn,
                 batch_size=0,
                 ):
        self.kernel_fn = kernel_fn
        self.batch_size = batch_size
        self.get = "nngp"
    
    def single_sum_kernel(self, x1, x2):
        N = x1.shape[0]
        input_shape = x1.shape[1:]
        
        def wrapper(x):
            x = x.reshape(input_shape)
            k = jnp.sum(self.kernel_fn(x, x2, self.get), axis=0) # sum(OxM) -> (M,)
            return k
        kernel = jnp.apply_along_axis(wrapper, 1, x1.reshape((N, -1)))
        return kernel # NxM
    
    def batched_single_sum_kernel(self, x1, x2):
        if self.batch_size < 1:
            return self.single_sum_kernel(x1, x2)
        
        N = x1.shape[0]
        M = x2.shape[0]
        kernel = []
        x1_start_indices = jnp.arange(0, N, self.batch_size)
        x1_end_indices = x1_start_indices + self.batch_size
        x2_start_indices = jnp.arange(0, M, self.batch_size)
        x2_end_indices = x2_start_indices + self.batch_size
        for x1_start, x1_end in zip(x1_start_indices, x1_end_indices):
            subkernel = []
            for x2_start, x2_end in zip(x2_start_indices, x2_end_indices):
                x1_end = min(x1_end, N)
                x2_end = min(x2_end, M)
                subkernel.append(self.single_sum_kernel(x1[x1_start:x1_end], x2[x2_start:x2_end]))
            kernel.append(jnp.concatenate(subkernel, axis=1))
        kernel = jnp.concatenate(kernel)
        return kernel
    
    def double_sum_kernel(self, x1, x2):        
        N = x1.shape[0]
        M = x2.shape[0]
        x1_shape = x1.shape[1:]
        x2_shape = x2.shape[1:]
        
        def along_x1(o1):
            o1 = o1.reshape(x1_shape)
            def along_x2(o2):
                o2 = o2.reshape(x2_shape)
                return jnp.sum(self.kernel_fn(o1, o2, self.get))
            k = jnp.apply_along_axis(along_x2, 1, x2.reshape((M, -1))).squeeze()
            return k

        kernel = jnp.apply_along_axis(along_x1, 1, x1.reshape((N, -1)))
        return kernel
    
    def batched_double_sum_kernel(self, x1, x2):
        if self.batch_size < 1:
            return self.double_sum_kernel(x1, x2)
        
        N = x1.shape[0]
        M = x2.shape[0]
        kernel = []
        x1_start_indices = jnp.arange(0, N, self.batch_size)
        x1_end_indices = x1_start_indices + self.batch_size
        x2_start_indices = jnp.arange(0, M, self.batch_size)
        x2_end_indices = x2_start_indices + self.batch_size
        for x1_start, x1_end in zip(x1_start_indices, x1_end_indices):
            subkernel = []
            for x2_start, x2_end in zip(x2_start_indices, x2_end_indices):
                x1_end = min(x1_end, N)
                x2_end = min(x2_end, M)
                subkernel.append(self.double_sum_kernel(x1[x1_start:x1_end], x2[x2_start:x2_end]))
            kernel.append(jnp.concatenate(subkernel, axis=1))
        kernel = jnp.concatenate(kernel)
        return kernel
    
    def double_sum_kernel_diag(self, data):
        N = data.shape[0]
        input_shape = data.shape[1:]
        
        def wrapper(x):
            x = x.reshape(input_shape)
            return jnp.sum(self.kernel_fn(x, None, self.get))
        diag = jnp.apply_along_axis(wrapper, 1, data.reshape((N, -1))).flatten()
        return diag
    
    def __call__(self, x1, x2, diff_domain=(False, False), diag=False):
        if diag:
            # assert jnp.allclose(x1, x2)
            if diff_domain == (True, True):
                return kernel_diagonal(self.kernel_fn, x1)
            return self.double_sum_kernel_diag(x1)
        else:
            if diff_domain == (True, True):
                return batch_kernel(self.kernel_fn, self.batch_size, x1, x2)
            if diff_domain == (False, True):
                return self.batched_single_sum_kernel(x1, x2)
            if diff_domain == (True, False):
                return self.batched_single_sum_kernel(x2, x1).T
            return self.batched_double_sum_kernel(x1, x2)

class iSNNGP(SNNGP):
    def __init__(self, 
                 model, 
                 model_params, 
                 train_data, 
                 train_augs,
                 inducing_points,
                 num_latent_gps, 
                 noise_variance=1, 
                 init_stds=None,
                 batch_size=0):
        super(iSNNGP, self).__init__(
            model, 
            model_params, 
            train_data, 
            inducing_points, 
            num_latent_gps, 
            noise_variance, 
            init_stds,
            batch_size
        )
        self.train_augs = train_augs
    
    def _init_kernel_fn(self, stds):
        kernel_fn = init_kernel_fn(self.model, stds, self.hyper_params)
        self.kernel_fn = SummationKernel(kernel_fn, self.batch_size)
    
    def _precomputation(self):
        x, _ = self.train_augs
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
    
    def _trace_term(self, AAT):
        """trace term
        """
        x, _ = self.train_augs
        
        trace_k = jnp.sum(self.kernel_fn(x, x, diff_domain=(False, False), diag=True)) / self.sigma_squared
        trace_q = jnp.trace(AAT)
        trace = trace_k - trace_q
        # logger.debug(f"trace: {trace}")
        return -0.5 * self.out_dim * trace
    
    def upper_bound(self):
        x, y = self.train_augs
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
    
    def log_marginal_likelihood(self):
        x, y = self.train_augs
        
        self._init_kernel_fn(self.stds)
        cov = self.kernel_fn(x, x)
        N = cov.shape[0]
        
        chol = scipy.linalg.cholesky(
                cov + self.sigma_squared * jnp.eye(N), lower=True)
        logger.debug(f"{chol.shape}, {y.shape}")
        alpha = scipy.linalg.cho_solve((chol, True), y)
        logger.debug(alpha.shape)
        ll = jnp.sum(-0.5 * jnp.einsum('ik,ik->k', y, alpha) -
                    jnp.sum(jnp.log(jnp.diag(chol))) - (N / 2.) * jnp.log(2. * jnp.pi))
        return ll
      
    def _training_loss(self, params):
        x, _ = self.train_augs
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
        # test = jnp.expand_dims(test, 1)
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
        kus = self.kernel_fn(self.inducing_points, test, diff_domain=(True, True)) # (M, N)
        
        tmp1 = scipy.linalg.solve_triangular(L, kus, lower=True) # (M, N)
        tmp2 = scipy.linalg.solve_triangular(LB, tmp1, lower=True) # (M, N)
        
        mean = tmp2.T @ c
        if diag:
            cov = self.kernel_fn(test, test, diff_domain=(True, True), diag=diag) + jnp.sum(jnp.square(tmp2), 0) - jnp.sum(jnp.square(tmp1), 0)
            cov = jnp.tile(cov[:, None], [1, self.num_latent_gps]) # (N, 1) -> (N, P)
        else:
            cov = self.kernel_fn(test, test, diff_domain=(True, True), diag=diag) + tmp2.T @ tmp2 - tmp1.T @ tmp1
            cov = jnp.tile(cov[None, ...], [self.num_latent_gps, 1, 1]) # (1, N, N) -> (P, N, N)

        return mean, cov


if __name__ == "__main__":
    pass
    