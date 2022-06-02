import jax.numpy as np
import neural_tangents as nt
from util import split_key

class GPNN():
    def __init__(self, model):
        self.model = model
        
    def fit(self, xs, ys):
        self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.model.kernel_fn, xs, ys, diag_reg=1e-4)
        
    def random_draw(self, xs, n: int=10) -> list:
        prior_draws = []
        for _ in range(n):
            _, net_key = split_key()
            _, params = self.model.init_fn(net_key, xs.shape)
            prior_draws += [self.model.apply_fn(params, xs)]
            
        self.model.params = params
        return prior_draws

    def compute_nngp_kernel(self, xs):
        return self.model.kernel_fn(xs, xs, 'nngp')

    def get_std(self, xs):
        kernel = self.compute_nngp_kernel(xs)
        std = np.sqrt(np.diag(kernel))
        return std

    def inference(self, xs, ntk=False):
        if ntk:
            mean, cov = self.predict_fn(x_test=xs, get="ntk", compute_cov=True)
        else:
            mean, cov = self.predict_fn(x_test=xs, get="nngp", compute_cov=True)
            
        return mean, cov

    def compute_loss(self, loss_fn, ys, t, xs=None):
        mean, cov = self.predict_fn(t=t, get='ntk', x_test=xs, compute_cov=True) # (ts, x_num, dim), (ts, x_num, x)
        mean = np.reshape(mean, mean.shape[:1] + (-1,)) # (ts, x_num)
        var = np.diagonal(cov, axis1=1, axis2=2) # (ts, x_num)
        ys = np.reshape(ys, (1, -1)) # (1, y_num)
        return loss_fn(mean, var, ys)