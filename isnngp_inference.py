import jax.numpy as np
from neural_tangents import stax
import logger as logging
from metrics import RMSE, accuracy
from isnngp import iSNNGP
from snngp_inference import select_inducing_points
from models import CNN, CNNShallow, FCN
from util import args_parser, init_random_state, check_divisibility
from data_loader import load_mnist_augments, load_mnist
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
        
    train, test = load_mnist(shuffle=False, flatten=flatten, one_hot=True, val_size=0)
    train_X, train_y = train
    N = train_X.shape[0]
    
    assert args.augment_X is not None and args.augment_y is not None
    augment_X, augment_y = load_mnist_augments(args.augment_X, args.augment_y, 
                                                flatten=flatten, one_hot=True)
    M = augment_X.shape[0]
    assert M % N == 0
    
    full_X = np.concatenate([train_X, augment_X])
    
    orbit_size = M // N
    augment_X_partitions = np.split(augment_X, orbit_size)
    augment_y_partitions = np.split(augment_y, orbit_size)
    augment_X_partitions = [np.expand_dims(x, 1) for x in augment_X_partitions]
    augment_X = np.concatenate(augment_X_partitions, 1)
    augments = (augment_X, augment_y_partitions[0])
    assert np.allclose(train_y, augments[1])

    # train_X = np.expand_dims(train_X, 1)
    # augment_X = np.expand_dims(augment_X, 1)
    # train_X = np.concatenate([train_X, augment_X], 1)
    # train = (train_X, train_y)
    
    check_divisibility(train[0], test[0], batch_size, device_count)
    return model, model_params, full_X, train, augments, test

def run():
    logger = logging.init_logger(log_level=logging.INFO)
    args = args_parser()
    init_random_state(0)
    model, model_params, full_X, train, augments, test = prepare_model_and_data(args)
    train_x, train_y = train
    test_x, test_y = test
    
    # stds = np.array([1.1, 0.15], dtype=np.float64)
    # noise_variance = 0.01
    stds = np.array([0.71294438, 0.06118174], dtype=np.float64)
    noise_variance = 0.01
    M = 500
    
    inducing_points, indices = select_inducing_points(args.select_method, full_X, M, model,stds, model_params)
    
    isnngp = iSNNGP(model=model, model_params=model_params, train_data=train, train_augs=augments, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
    
    # success, _, _ = isnngp.optimize(compile=True, disp=False)
    # while not success:
    #     logger.info("Retry...")
    #     inducing_points, indices = select_inducing_points(args.select_method, train_x, args.num_inducing_points, model,stds, model_params)
    #     isnngp = iSNNGP(model=model, model_params=model_params, train_data=train, train_augs=augments, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
    #     success, stds, noise_variance = isnngp.optimize(compile=True, disp=False)
    
    lml = isnngp.log_marginal_likelihood()
    logger.info(f"LML: {lml:.4f}")
    elbo = isnngp.lower_bound()
    logger.info(f"ELBO: {elbo:.4f}")
    eubo = isnngp.upper_bound()
    logger.info(f"EUBO: {eubo:.4f}")
    mean, var = isnngp.predict(test_x, diag=True)
    logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
    acc = accuracy(mean, test_y)
    logger.info(f"Accuracy: {acc:.2%}")
    rmse = RMSE(mean, test_y)
    logger.info(f"Loss: {rmse:.4f}")
    
if __name__ == "__main__":
    run()
