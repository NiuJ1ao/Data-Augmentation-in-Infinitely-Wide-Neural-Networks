import jax.numpy as np
import matplotlib.pyplot as plt

import logger as logging
from metrics import RMSE, accuracy
from snngp import SNNGP
from snngp_inference import prepare_model_and_data
from inducing_points import select_inducing_points
from util import args_parser, init_random_state
from snngp import init_kernel_fn
from inducing_points import greedy_variance_generator
from tqdm import tqdm

logger = logging.init_logger(log_level=logging.INFO)
from jax.config import config

config.update("jax_enable_x64", True)

def run():
    args = args_parser()
    init_random_state(0)
    model, model_params, train, test = prepare_model_and_data(args)
    train_x, train_y = train
    test_x, test_y = test
    stds = np.array([1.11264191, 0.16673347], dtype=np.float64)
    noise_variance = 0.011984670523132117
    kernel_fn = init_kernel_fn(model, stds, model_params)
    inducing_generator = greedy_variance_generator(train_x, 20000, kernel_fn)
    Zs = []
    nums_inducing_points = [10000] # np.logspace(np.log10(100), np.log10(5000), 20)
    prev_m = 0
    for m in nums_inducing_points:
        m = round(m)
        if m == prev_m:
            continue
        logger.info(f"\n----Number of inducing point: {m}----")
        
        # for _ in tqdm(range(m-prev_m)):
        #     Z, _ = next(inducing_generator)
        #     Zs.append(Z)
        # inducing_points = np.vstack(Zs)
        inducing_points, _ = select_inducing_points(args.select_method, train_x, m, model, stds, model_params)
        np.save(f"/vol/bitbucket/yn621/data/inducing_points_{m}", inducing_points)
        snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
        try:
            success, stds, noise_variance = snngp.optimize(compile=True, disp=False)
        except Exception as e:
            logger.warning(f"Fail to optimize: {e}")
        
        lml = snngp.log_marginal_likelihood()
        logger.info(f"LML: {lml}")
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
        prev_m = m
        
    for m in np.arange(11000, 21000, 1000):
        m = round(m)
        if m == prev_m:
            continue
        logger.info(f"\n----Number of inducing point: {m}----")
        
        for _ in range(m-prev_m):
            Z, _ = next(inducing_generator)
            Zs.append(Z)
        inducing_points = np.vstack(Zs)
        # inducing_points, _ = select_inducing_points(args.select_method, train_x, m, model, stds, model_params)
        np.save(f"/vol/bitbucket/yn621/data/inducing_points_{m}", inducing_points)
        snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
        
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
        prev_m = m

    
if __name__ == "__main__":
    run()
