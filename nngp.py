import time
import jax.numpy as np
import neural_tangents as nt
from logger import get_logger
logger = get_logger()

class NNGP():
    def __init__(self, model):
        self.model = model
        # self.kernel_fn = nt.empirical_nngp_fn(self.model.apply_fn)
        
    def fit(self, x, y):
        logger.info("Inferencing...")
        start = time.time()
        # k_train_train = self.kernel_fn(x, x, self.model.params)
        # self.predict_fn = nt.predict.gradient_descent_mse(k_train_train=k_train_train, y_train=y, diag_reg=1e-4)
        # self.predict_fn = nt.predict.gp_inference(k_train_train=k_train_train, y_train=y, diag_reg=1e-4)
        # self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.kernel_fn, x, y, diag_reg=1e-4, params=self.model.params)
        self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.model.kernel_fn, x, y, diag_reg=1e-4)
        elapsed_time = time.time() - start
        logger.info(f"Inference finished in {elapsed_time:0.2f} sec")
        
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
        logger.info("Predicting...")
        start = time.time()
        if ntk:
            mean, cov = self.predict_fn(x_test=x_test, get="ntk", compute_cov=True)
        else:
            # fx_train_0 = self.model.predict(x_train)
            # fx_test_0 = self.model.predict(x_test)
            # k_test_train = self.kernel_fn(x_test, x_train, self.model.params)
            # k_test_test = self.kernel_fn(x_test, x_test, self.model.params)
            # fx_train, fx_test = self.predict_fn(fx_train_0=fx_train_0, fx_test_0=fx_test_0, k_test_train=k_test_train)
            
            # gaussian = self.predict_fn(get="nngp", k_test_train = k_test_train, k_test_test=k_test_test)
            # mean = gaussian.mean
            # cov = gaussian.covariance
            
            mean, cov = self.predict_fn(x_test=x_test, get="nngp", compute_cov=True)
            
        elapsed_time = time.time() - start
        logger.info(f"Prediction finished in {elapsed_time:0.2f} sec")
        return mean, cov

    def compute_loss(self, loss_fn, ys, t, xs=None):
        mean, cov = self.predict_fn(t=t, get='nngp', x_test=xs, compute_cov=True) # (ts, x_num, dim), (ts, x_num, x)
        mean = np.reshape(mean, mean.shape[:1] + (-1,)) # (ts, x_num)
        var = np.diagonal(cov, axis1=1, axis2=2) # (ts, x_num)
        ys = np.reshape(ys, (1, -1)) # (1, y_num)
        return loss_fn(mean, var, ys)