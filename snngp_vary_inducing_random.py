import jax.numpy as np
import matplotlib.pyplot as plt

import logger as logging
from metrics import RMSE, accuracy
from snngp import SNNGP
from snngp_inference import prepare_model_and_data, select_inducing_points
from util import args_parser, init_random_state

logger = logging.init_logger(log_level=logging.INFO)
from jax.config import config

config.update("jax_enable_x64", True)

def run():
    args = args_parser()
    init_random_state(0)
    model, model_params, train, test = prepare_model_and_data(args)
    train_x, train_y = train
    test_x, test_y = test
    stds = np.array([1.5, 0.1], dtype=np.float64)
    noise_variance = 0.0001
    nums_inducing_points = [1000, 3000] # np.logspace(np.log10(100), np.log10(5000), 20)
    prev_m = 0
    for m in nums_inducing_points:
        m = round(m)
        if m == prev_m:
            continue
        prev_m = m
        logger.info(f"\n----Number of inducing point: {m}----")
        
        inducing_points, _ = select_inducing_points(args.select_method, train_x, m)
        snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
        try:
            success, stds, noise_variance = snngp.optimize(compile=True, disp=False)
            while not success:
                logger.info("Retry...")
                inducing_points, _ = select_inducing_points(args.select_method, train_x, m)
                snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
                success, stds, noise_variance = snngp.optimize(compile=True, disp=False)
        except Exception as e:
            logger.warning(f"Fail to optimize: {e}")
        
        # lml = snngp.log_marginal_likelihood()
        # logger.info(f"LML: {lml}")
        elbo = snngp.lower_bound()
        logger.info(f"ELBO: {elbo}")
        eubo = snngp.upper_bound()
        logger.info(f"EUBO: {eubo}")
        mean, var = snngp.predict(test_x, diag=True)
        logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
        loss = RMSE(mean, test_y)
        logger.info(f"Loss: {loss}")
        acc = accuracy(mean, test_y)
        logger.info(f"Accuracy: {acc:.2%}")
        
        del inducing_points
        del snngp
    
    stds = np.array([1.08825866, 0.13723203], dtype=np.float64)
    noise_variance = 0.014345011621388796
    for m in np.arange(5000, 21000, 1000):
        m = round(m)
        if m == prev_m:
            continue
        prev_m = m
        logger.info(f"\n----Number of inducing point: {m}----")
        
        inducing_points, _ = select_inducing_points(args.select_method, train_x, m)
        snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
        elbo = snngp.lower_bound()
        logger.info(f"ELBO: {elbo}")
        eubo = snngp.upper_bound()
        logger.info(f"EUBO: {eubo}")
        mean, var = snngp.predict(test_x, diag=True)
        logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
        loss = RMSE(mean, test_y)
        logger.info(f"Loss: {loss}")
        acc = accuracy(mean, test_y)
        logger.info(f"Accuracy: {acc:.2%}")
        
        del inducing_points
        del snngp
    
if __name__ == "__main__":
    run()
