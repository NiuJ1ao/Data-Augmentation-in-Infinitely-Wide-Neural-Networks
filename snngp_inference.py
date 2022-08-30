from collections import Counter
import jax.numpy as np
from neural_tangents import stax

import logger as logging
from data_loader import load_mnist, load_mnist_augments, load_mnist_full
from inducing_points import greedy_variance, random_select, first_n
from metrics import RMSE, accuracy
from models import CNN, CNNShallow, FCN
from snngp import SNNGP, init_kernel_fn
from util import args_parser, check_divisibility, init_random_state
from jax.config import config
logger = logging.get_logger()
config.update("jax_enable_x64", True)

def prepare_model_and_data(args):
    batch_size = 0
    device_count = args.device_count
    
    if args.model == 'cnn':
        model = CNNShallow
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
            out_dim=10,
            nonlinearity=stax.Relu
            )
        flatten = True
        
    train, test = load_mnist_full(shuffle=False, flatten=flatten, one_hot=True, val_size=0)
    if args.augment_X != None:
        augment_X, augment_y = load_mnist_augments(args.augment_X, args.augment_y, 
                                                   flatten=flatten, one_hot=True)
        train_X, train_y = train
        train_X = np.concatenate([train_X, augment_X])
        train_y = np.concatenate([train_y, augment_y])
        train = (train_X, train_y)
        logger.info(f"Training data after augmentation: {train_X.shape}, {train_y.shape}")
    check_divisibility(train[0], test[0], batch_size, device_count)
    return model, model_params, train, test

def select_inducing_points(method, train, M, model=None, stds=None, model_params=None):
    if method == "random":
        inducing_points, indices = random_select(train, M)
    elif method == "first":
        inducing_points, indices = first_n(train, M)
    elif method == "greedy":
        kernel_fn = init_kernel_fn(model, stds, model_params)
        inducing_points, indices = greedy_variance(train, M, kernel_fn)
    logger.info(f"inducing_points shape: {inducing_points.shape}")
    return inducing_points, indices

def run():
    logger = logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    init_random_state(0)
    model, model_params, train, test = prepare_model_and_data(args)
    train_x, train_y = train
    test_x, test_y = test
    
    # stds = np.array([1.5, 0.15], dtype=np.float64)
    # noise_variance = 0.0001
    # stds = np.array([1.5, 0.1], dtype=np.float64)
    # noise_variance = 0.0001
    stds = np.array([1.11264191, 0.16673347], dtype=np.float64)
    noise_variance = 0.011984670523132117
    # stds = np.array([1., 1.], dtype=np.float64)
    # noise_variance = 0.01
    
    # inducing_points, indices = select_inducing_points(args.select_method, train_x, args.num_inducing_points, model,stds, model_params)
    # inducing_labels = np.argmax(train_y[indices], axis=-1)
    # logger.info(f"Inducing points stats: {Counter(inducing_labels.astype(int).tolist())}")
    inducing_points = np.load(f"/vol/bitbucket/yn621/data/inducing_points_{args.num_inducing_points}.npy")
    logger.info(f"inducing_points shape: {inducing_points.shape}")
    
    snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, noise_variance=noise_variance, init_stds=stds, batch_size=args.batch_size)
    
    # success, _, _ = snngp.optimize(compile=True, disp=False)
    # while not success:
    #     logger.info("Retry...")
    #     inducing_points, _ = select_inducing_points(args.select_method, train_x, args.num_inducing_points, model,stds, model_params)
    #     snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
    #     success, stds, noise_variance = snngp.optimize(compile=True, disp=False)
    
    # lml = snngp.log_marginal_likelihood()
    # logger.info(f"LML: {lml:.4f}")
    elbo = snngp.lower_bound()
    logger.info(f"ELBO: {elbo:.4f}")
    eubo = snngp.upper_bound()
    logger.info(f"EUBO: {eubo:.4f}")
    mean, var = snngp.predict(test_x, diag=True)
    logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
    acc = accuracy(mean, test_y)
    logger.info(f"Accuracy: {acc:.2%}")
    rmse = RMSE(mean, test_y)
    logger.info(f"Loss: {rmse:.4f}")
    
if __name__ == "__main__":
    run()
