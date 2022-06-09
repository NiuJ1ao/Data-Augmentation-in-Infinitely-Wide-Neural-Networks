import itertools
import time
from jax import jit, grad, vmap
import jax.numpy as np
from util import PRNGKey, minibatch, split_key
from tqdm import tqdm
from logger import get_logger
logger = get_logger()
from copy import deepcopy

class Trainer():
    
    def __init__(self, 
                 model, 
                 epochs, 
                 batch_size,
                 optimizer, 
                 loss):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt_init, opt_update, self.get_params = optimizer
        
        self.opt_update = jit(opt_update)
        self.loss = jit(loss)
        self.grad_loss = jit(lambda state, x, y: grad(self.loss)(self.get_params(state), x, y))
        
    def fit(self, train, test, 
            metric=None,
            init_params=False):
        
        # batch training data
        num_complete_batches, leftover = divmod(train[0].shape[0], self.batch_size)
        num_batches = num_complete_batches + bool(leftover)
        train_batches = minibatch(*train, batch_size=self.batch_size, num_batches=num_batches)

        # initialise model if not yet initialised
        if self.model.params != None and not init_params:
            opt_state = self.opt_init(self.model.params)
        else:
            _, net_key = split_key()
            _, init_params = self.model.init_fn(net_key, train[0].shape)
            opt_state = self.opt_init(init_params)

        # train step
        itercount = itertools.count()
        total_steps = self.epochs * num_batches
        pbar = tqdm(total=total_steps)
        for _ in range(self.epochs):
            for _ in range(num_batches):
                opt_state = self.opt_update(next(itercount), self.grad_loss(opt_state, *next(train_batches)), opt_state)
                pbar.update(1)

            step_params = self.get_params(opt_state)
            if metric != None:
                train_m = metric(step_params, *train).item()
                test_m = metric(step_params, *test).item()
                logger.info(f"train acc: {train_m}, test acc: {test_m}")
            
        return self.get_params(opt_state)
    
    def ensemble_fit(self, train, test, num_models):
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
        
        if num_models == 1:
            return _fit(PRNGKey.key)
        
        ensemble_key = split_key(num_models)
        params, train_loss, test_loss = vmap(_fit)(ensemble_key)
        return params, train_loss, test_loss