import time
import jax.numpy as np
import neural_tangents as nt
from jax import scipy
from logger import get_logger
from util import batch_kernel
from tqdm import tqdm
logger = get_logger()

def compute_full_kernel(kernel_fn, batch_size, x1, x2):
    if batch_size < 1:
        return kernel_fn(x1, x2, "nngp")
    
    N = x1.shape[0]
    M = x2.shape[0]
    x1_start_indices = np.arange(0, N, batch_size)
    x1_end_indices = x1_start_indices + batch_size
    x2_start_indices = np.arange(0, M, batch_size)
    x2_end_indices = x1_start_indices + batch_size
    p_bar = tqdm(total=len(x1_start_indices)*len(x2_start_indices), desc="Compute and save sub-kernels")
    for x1_start, x1_end in zip(x1_start_indices, x1_end_indices):
        for x2_start, x2_end in zip(x2_start_indices, x2_end_indices):
            x1_end = min(x1_end, N)
            x2_end = min(x2_end, M)
            np.save(f"/vol/bitbucket/yn621/data/mnist_kernel_{x1_start}-{x1_end}_{x2_start}-{x2_end}", kernel_fn(x1[x1_start:x1_end], x2[x2_start:x2_end], "nngp"))
            p_bar.update(1)
    
    p_bar = tqdm(total=len(x1_start_indices)*len(x2_start_indices), desc="Load sub-kernels")
    kernel = []
    for x1_start, x1_end in zip(x1_start_indices, x1_end_indices):
        subkernel = []
        for x2_start, x2_end in zip(x2_start_indices, x2_end_indices):
            x1_end = min(x1_end, N)
            x2_end = min(x2_end, M)     
            subkernel.append(np.load(f"/vol/bitbucket/yn621/data/mnist_kernel_{x1_start}-{x1_end}_{x2_start}-{x2_end}.npy"))
            p_bar.update(1)
        kernel.append(np.concatenate(subkernel, axis=1))
    return np.concatenate(kernel)


class NNGP():
    def __init__(self, model, batch_size, train, sigma_squared=1):
        self.model = model
        self.batch_size = batch_size
        self.train = train
        self.sigma_squared = sigma_squared
        self.chol = self._compute_chol()
        
    def _compute_chol(self):
        train_x, _ = self.train
        N = train_x.shape[0]
        cov = compute_full_kernel(self.model.kernel_fn, self.batch_size, train_x, train_x) + self.sigma_squared * np.eye(N)
        chol = np.linalg.cholesky(cov)
        return chol
        
    def fit(self):
        logger.info("Fitting...")
        start = time.time()
        train_x, train_y = self.train
        self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.model.kernel_fn, train_x, train_y, diag_reg=1e-4)
        elapsed_time = time.time() - start
        logger.info(f"Fit finished in {elapsed_time:0.2f} sec")
        
    def random_draw(self, xs, n: int=10) -> list:
        prior_draws = []
        for _ in range(n):
            self.model.init_params(xs.shape, force_init=True)
            prior_draws += [self.model.predict(xs)]
        
        return prior_draws

    def compute_nngp_kernel(self, x):
        return self.model.kernel_fn(x, x, 'nngp')

    def get_std(self, x):
        kernel = self.compute_nngp_kernel(x)
        std = np.sqrt(np.diag(kernel))
        return std

    def inference(self, x_test, ntk=False):
        logger.info("Inferencing...")
        start = time.time()
        if ntk:
            mean, cov = self.predict_fn(x_test=x_test, get="ntk", compute_cov=True)
        else:
            mean, cov = self.predict_fn(x_test=x_test, get="nngp", compute_cov=True)
        elapsed_time = time.time() - start
        logger.info(f"Inference finished in {elapsed_time:0.2f} sec")
        return mean, cov
    
    def log_marginal_likelihood(self):
        train_x, train_y = self.train
        N = train_x.shape[0]
        chol = self.chol
        alpha = scipy.linalg.cho_solve((chol, True), train_y)

        ll = np.sum(-0.5 * np.einsum('ik,ik->k', train_y, alpha) -
                    np.sum(np.log(np.diag(chol))) - (N / 2.) * np.log(2. * np.pi))
        return ll
    
    def predict_fn(self, train, test_x):
        train_x, train_y = train
        M = test_x.shape[0]
        ksx = batch_kernel(self.model.kernel_fn, self.batch_size, test_x, train_x)
        logger.debug(f"ksx: {ksx.shape}")
        kss = batch_kernel(self.model.kernel_fn, self.batch_size, test_x, test_x)
        logger.debug(f"kss: {kss.shape}")
        chol = self.chol
        mean = ksx @ scipy.linalg.cho_solve((chol, True), train_y)
        cov = kss + self.sigma_squared * np.eye(M) -  ksx @ scipy.linalg.cho_solve((chol, True), ksx.T)
        logger.debug(f"{mean.shape}, {cov.shape}")
        return mean, cov

    def compute_loss(self, loss_fn, ys, t, xs=None):
        mean, cov = self.predict_fn(t=t, get='nngp', x_test=xs, compute_cov=True) # (ts, x_num, dim), (ts, x_num, x)
        mean = np.reshape(mean, mean.shape[:1] + (-1,)) # (ts, x_num)
        var = np.diagonal(cov, axis1=1, axis2=2) # (ts, x_num)
        ys = np.reshape(ys, (1, -1)) # (1, y_num)
        return loss_fn(mean, var, ys)