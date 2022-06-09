import time
import jax.numpy as np
import neural_tangents as nt
from util import split_key
from logger import get_logger
logger = get_logger()

class NNGP():
    def __init__(self, model):
        self.model = model
        
    def fit(self, x, y):
        logger.info("Inferencing...")
        start = time.time()
        self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.model.kernel_fn, x, y, diag_reg=1e-4)
        elapsed_time = time.time() - start
        logger.info(f"Inference finished in {elapsed_time:0.2f} sec")
        
    def random_draw(self, xs, n: int=10) -> list:
        prior_draws = []
        for _ in range(n):
            _, net_key = split_key()
            self.model.init_params(net_key, xs.shape)
            prior_draws += [self.model.predict(xs)]
        
        return prior_draws

    def compute_nngp_kernel(self, x):
        return self.model.kernel_fn(x, x, 'nngp')

    def get_std(self, x):
        kernel = self.compute_nngp_kernel(x)
        std = np.sqrt(np.diag(kernel))
        return std

    def inference(self, x, ntk=False):
        logger.info("Predicting...")
        start = time.time()
        if ntk:
            mean, cov = self.predict_fn(x_test=x, get="ntk", compute_cov=True)
        else:
            mean, cov = self.predict_fn(x_test=x, get="nngp", compute_cov=True)
        elapsed_time = time.time() - start
        logger.info(f"Prediction finished in {elapsed_time:0.2f} sec")
        return mean, cov

    def compute_loss(self, loss_fn, ys, t, xs=None):
        mean, cov = self.predict_fn(t=t, get='ntk', x_test=xs, compute_cov=True) # (ts, x_num, dim), (ts, x_num, x)
        mean = np.reshape(mean, mean.shape[:1] + (-1,)) # (ts, x_num)
        var = np.diagonal(cov, axis1=1, axis2=2) # (ts, x_num)
        ys = np.reshape(ys, (1, -1)) # (1, y_num)
        return loss_fn(mean, var, ys)