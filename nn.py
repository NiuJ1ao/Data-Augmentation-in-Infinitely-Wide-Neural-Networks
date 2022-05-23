from jax import jit, grad
from util import split_key

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
            
        return opt_state, train_losses, test_losses