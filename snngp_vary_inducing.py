import jax.numpy as np
import matplotlib.pyplot as plt

import logger as logging
from metrics import accuracy
from snngp import SNNGP
from snngp_inference import prepare_model_and_data, select_inducing_points
from util import args_parser, init_random_state

logger = logging.init_logger(log_level=logging.INFO)
from jax.config import config

config.update("jax_enable_x64", True)


def run():
    args = args_parser()
    init_random_state(0)
    model, model_params, train, val, test = prepare_model_and_data(args)
    train_x, train_y = train
    test_x, test_y = test
    stds = np.array([1.5, 0.05], dtype=np.float64)
    lbs, ubs, sharps, accs = [], [], [], []
    nums_inducing_points = np.arange(3150, 10050, 50)
    for m in nums_inducing_points:
        logger.info(f"\n----Number of inducing point: {m}----")
        inducing_points = select_inducing_points(args.select_method, train_x, m, model, stds, model_params)
                
        snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size)
        
        # snngp.optimize()

        elbo = snngp.lower_bound()
        logger.info(f"ELBO: {elbo}")
        eubo = snngp.upper_bound()
        logger.info(f"EUBO: {eubo}")
        mean, var = snngp.predict(test_x, diag=True)
        logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
        sharpness = np.var(var)
        logger.info(f"Sharpness: {sharpness}")
        acc = accuracy(mean, test_y)
        logger.info(f"Accuracy: {acc:.2%}")
        
        lbs.append(elbo)
        ubs.append(eubo)
        sharps.append(sharpness)
        accs.append(acc)
    
    plt.plot(nums_inducing_points, lbs)
    plt.savefig(f"figures/vary_inducing_{args.select_method}_{args.model}_lbs")
    plt.close()
    plt.plot(nums_inducing_points, ubs)
    plt.savefig(f"figures/vary_inducing_{args.select_method}_{args.model}_ubs")
    plt.close()
    plt.plot(nums_inducing_points, sharps)
    plt.savefig(f"figures/vary_inducing_{args.select_method}_{args.model}_sharps")
    plt.close()
    plt.plot(nums_inducing_points, accs)
    plt.savefig(f"figures/vary_inducing_{args.select_method}_{args.model}_accs")
    plt.close()
    
if __name__ == "__main__":
    run()
