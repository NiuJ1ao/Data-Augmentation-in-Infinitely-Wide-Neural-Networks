import itertools
from jax import jit, grad, vmap
import jax.numpy as np
from util import PRNGKey, compute_num_batches, minibatch, split_key
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
        
    def fit(self, train, val, 
            metric=None,
            ):
        
        # batch training data
        num_batches = compute_num_batches(train[0].shape[0], self.batch_size)
        logger.debug(f"Num of batches: {num_batches}")
        train_batches = minibatch(*train, batch_size=self.batch_size, num_batches=num_batches)
        val_x, val_y = val
        
        # # initialise model if not yet initialised
        # if padding:
        #     if train[0].shape[1] < 224 or train[0].shape[2] < 224:
        #         input_shape = (self.batch_size, 224, 224, 1)
        #     else:
        #         input_shape = (self.batch_size,) + train[0].shape[1:]
        # else:
        #     input_shape = (self.batch_size,) + train[0].shape[1:]
        input_shape = (self.batch_size,) + train[0].shape[1:]
        logger.debug(f"input shape: {input_shape}")
        self.model.init_params(input_shape)
        opt_state = self.opt_init(self.model.params)
        
        train_losses = []
        val_losses = []
        train_evals = []
        val_evals = []

        # train step
        itercount = itertools.count()
        total_steps = self.epochs * num_batches
        pbar = tqdm(total=total_steps)
        for i in range(self.epochs):
            # train
            for _ in range(num_batches):
                train_x, train_y = next(train_batches)
                opt_state = self.opt_update(next(itercount), self.grad_loss(opt_state, train_x, train_y), opt_state)
                self.model.update_params(self.get_params(opt_state))
                pbar.update(1)
            
            # eval
            epoch_train_loss, epoch_train_eval = 0, 0
            for _ in range(num_batches):
                train_x, train_y = next(train_batches)
                train_loss = self.loss(self.get_params(opt_state), train_x, train_y)
                epoch_train_loss += train_loss
                if metric != None:
                    train_preds = self.model.predict(train_x)
                    epoch_train_eval += metric(train_preds, train_y)

            train_losses += [epoch_train_loss / num_batches]
            val_loss = self.loss(self.get_params(opt_state), val_x, val_y)
            val_losses += [val_loss]
            
            if metric != None:
                epoch_train_eval = epoch_train_eval / num_batches
                epoch_val_eval = metric(self.model.predict(val_x), val_y)
                train_evals += [epoch_train_eval]
                val_evals += [epoch_val_eval]
                logger.info(f"Epoch {i}; Train acc: {epoch_train_eval:.2%}; Val acc: {epoch_val_eval:.2%}")
                
        if metric != None:
            return self.get_params(opt_state), train_losses, val_losses, train_evals, val_evals
                  
        return self.get_params(opt_state), train_losses, val_losses
          
          
    def fit_small(self, train, val, 
        metric=None,
        ):
    
        # batch training data
        num_batches = compute_num_batches(train[0].shape[0], self.batch_size)
        train_batches = minibatch(*train, batch_size=self.batch_size, num_batches=num_batches)
        
        train_losses, test_losses = [], []
        
        # initialise model if not yet initialised
        self.model.init_params(train[0].shape)
        opt_state = self.opt_init(self.model.params)

        # train step
        itercount = itertools.count()
        total_steps = self.epochs * num_batches
        pbar = tqdm(total=total_steps)
        for _ in range(self.epochs):
            for _ in range(num_batches):
                opt_state = self.opt_update(next(itercount), self.grad_loss(opt_state, *next(train_batches)), opt_state)
                self.model.update_params(self.get_params(opt_state))
                pbar.update(1)
                
            train_losses += [self.loss(self.get_params(opt_state), *train)]
            test_losses += [self.loss(self.get_params(opt_state), *val)]
            
        
            if metric != None:
                train_preds = self.model.predict(train[0])
                train_m = metric(train_preds, train[1]).item()
                val_preds = self.model.predict(val[0])
                val_m = metric(val_preds, val[1]).item()
                logger.info(f"train acc: {train_m:.4%}, val acc: {val_m:.4%}")
        
        return self.get_params(opt_state), train_losses, test_losses
                
    
    def fit_ensemble(self, train, test, num_models):
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