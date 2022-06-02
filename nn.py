from jax import jit, grad, vmap
import jax.numpy as np
from util import PRNGKey, split_key

class Trainer():
    
    def __init__(self, model, training_steps, optimizer, loss):
        self.model = model
        self.training_steps = training_steps
        self.opt_init, opt_update, self.get_params = optimizer
        
        self.opt_update = jit(opt_update)
        self.loss = jit(loss)
        self.grad_loss = jit(lambda state, x, y: grad(self.loss)(self.get_params(state), x, y))
        
    def fit(self, train, test):
        train_losses = []
        test_losses = []

        opt_state = self.opt_init(self.model.params)

        for i in range(self.training_steps):
            opt_state = self.opt_update(i, self.grad_loss(opt_state, *train), opt_state)

            train_losses += [self.loss(self.get_params(opt_state), *train)]
            test_losses += [self.loss(self.get_params(opt_state), *test)]

        return self.get_params(opt_state), train_losses, test_losses
    
    def ensemble_fit(self, train, test, size):
        def _fit(key):
            train_losses = []
            test_losses = []

            _, params = self.model.init_fn(key, train[0].shape)
            opt_state = self.opt_init(params)

            for i in range(self.training_steps):
                train_losses += [np.reshape(self.loss(self.get_params(opt_state), *train),  (1,))]
                test_losses += [np.reshape(self.loss(self.get_params(opt_state), *test), (1,))]
                opt_state = self.opt_update(i, self.grad_loss(opt_state, *train), opt_state)

            train_losses = np.concatenate(train_losses)
            test_losses = np.concatenate(test_losses)
            return self.get_params(opt_state), train_losses, test_losses
        
        if size == 1:
            return _fit(PRNGKey.key)
        
        ensemble_key = split_key(size)
        params, train_loss, test_loss = vmap(_fit)(ensemble_key)
        return params, train_loss, test_loss