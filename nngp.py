import time
import jax.numpy as np
import neural_tangents as nt
from jax import scipy
from logger import get_logger
logger = get_logger()


class NNGP():
    def __init__(self, model, train, sigma_squared=1):
        self.model = model
        self.train = train
        self.sigma_squared = sigma_squared
        
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
        cov = self.model.kernel_fn(train_x, None, "nngp") + self.sigma_squared * np.eye(N)
        chol = np.linalg.cholesky(cov)
        alpha = scipy.linalg.cho_solve((chol, True), train_y)

        ll = np.sum(-0.5 * np.einsum('ik,ik->k', train_y, alpha) -
                    np.sum(np.log(np.diag(chol))) - (N / 2.) * np.log(2. * np.pi))
        return ll
    
    def predict_fn_(self, test_x):
        train_x, train_y = self.train
        N = train_x.shape[0]
        M = test_x.shape[0]
        ksx = self.model.kernel_fn(test_x, train_x, "nngp")
        logger.debug(f"ksx: {ksx.shape}")
        kss = self.model.kernel_fn(test_x, test_x, "nngp")
        logger.debug(f"kss: {kss.shape}")
        cov = self.model.kernel_fn(train_x, None, "nngp") + self.sigma_squared * np.eye(N)
        chol = np.linalg.cholesky(cov)
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