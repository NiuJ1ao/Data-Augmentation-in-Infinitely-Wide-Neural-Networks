import jax.numpy as np
from neural_tangents import stax

import logger as logging
from data_loader import load_mnist
from inducing_points import greedy_variance, random_select, first_n
from metrics import accuracy
from models import CNN, FCN
from snngp import SNNGP, init_kernel_fn
from util import args_parser, check_divisibility, init_random_state
from jax.config import config
logger = logging.get_logger()
config.update("jax_enable_x64", True)

def prepare_model_and_data(args):
    batch_size = 0
    device_count = args.device_count
    
    if args.model == 'cnn':
        model = CNN
        model_params = dict(
            kernel_batch_size=batch_size, 
            device_count=device_count, 
            num_classes=10
            )
        flatten = False
    elif args.model == 'fcn':
        model = FCN
        model_params = dict(
            kernel_batch_size=batch_size, 
            device_count=device_count, 
            num_layers=2,
            nonlinearity=stax.Relu
            )
        flatten = True
        
    train, val, test = load_mnist(flatten=flatten, one_hot=True)
    check_divisibility(train[0], test[0], batch_size, device_count)
    return model, model_params, train, val, test

def select_inducing_points(method, train, M, model=None, stds=None, model_params=None):
    if method == "random":
        inducing_points = random_select(train, M)
    elif method == "first":
        inducing_points = first_n(train, M)
    elif method == "greedy":
        kernel_fn = init_kernel_fn(model, stds, model_params)
        inducing_points = greedy_variance(train, M, kernel_fn)
    logger.info(f"inducing_points shape: {inducing_points.shape}")
    return inducing_points

def run():
    logger = logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    init_random_state(0)
    model, model_params, train, val, test = prepare_model_and_data(args)
    train_x, train_y = train
    val_x, val_y = val
    test_x, test_y = test
    
    stds = np.array([1., 1.], dtype=np.float64)
    
    inducing_points = select_inducing_points(args.select_method, train_x, args.num_inducing_points, model,stds, model_params)
    
    snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size)
    
    elbo = snngp.lower_bound()
    logger.info(f"ELBO: {elbo}")
    eubo = snngp.upper_bound()
    logger.info(f"EUBO: {eubo}")
    mean, var = snngp.predict(val_x, diag=True)
    acc = accuracy(mean, val_y)
    logger.info(f"Accuracy: {acc:.2%}")
    mean, var = snngp.predict(test_x, diag=True)
    logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
    sharpness = np.var(var)
    logger.info(f"Sharpness: {sharpness}")
    acc = accuracy(mean, test_y)
    logger.info(f"Accuracy: {acc:.2%}")
    
    # snngp.optimize()
    
    # elbo = snngp.lower_bound()
    # logger.info(f"elbo: {elbo}")
    # eubo = snngp.upper_bound()
    # logger.info(f"eubo: {eubo}")
    # mean, var = snngp.predict(test_x, diag=True)
    # logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
    # sharpness = np.var(var)
    # logger.info(f"Sharpness: {sharpness}")
    # acc = accuracy(mean, test_y)
    # logger.info(f"Accuracy: {acc:.2%}")
    
if __name__ == "__main__":
    run()
