import jax.numpy as np

import logger as logging
import matplotlib.pyplot as plt
from inducing_points import greedy_variance
from metrics import RMSE, accuracy
from snngp import SNNGP, init_kernel_fn
from snngp_inference import prepare_model_and_data
from util import args_parser, init_random_state
logger = logging.init_logger(log_level=logging.DEBUG)
from jax.config import config
config.update("jax_enable_x64", True)

def run():
    args = args_parser()
    init_random_state(0)
    model, model_params, train, test = prepare_model_and_data(args)
    train_x, _ = train
    test_x, test_y = test
    
    stds = np.array([1., 1.], dtype=np.float64)
    accs, lbs, ubs, sharps, rmses = [], [], [], [], []
    step = 0
    elbo = 0
    while True:
        logger.info(f"\nstds: {stds}")
        prev_stds = stds
        prev_elbo = elbo
        
        kernel_fn = init_kernel_fn(model, stds, model_params)
        inducing_points = greedy_variance(train_x, args.num_inducing_points, kernel_fn)
        logger.info(f"inducing_points shape: {inducing_points.shape}")

        snngp = SNNGP(model=model, hyper_params=model_params, train_data=train, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds)
        
        stds = snngp.optimize()
        
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
        rmse = RMSE(mean, test_y)
        logger.info(f"RMSE: {rmse}")
        
        lbs.append(elbo)
        ubs.append(eubo)
        sharps.append(sharpness)
        accs.append(acc)
        rmses.append(rmse)
        step += 1

        # terminate condition
        if np.allclose(stds, prev_stds):
            logger.info(f"Terminating...")
            break
        
        if step == 10:
            logger.info(f"Reach the maximum iteration.")
            break
        
        # if elbo < prev_elbo or np.isclose(elbo, prev_elbo):
        #     logger.info(f"Terminating...")
        #     break
            
    
    x_axis = np.arange(step)
    plt.plot(x_axis, lbs)
    plt.savefig(f"figures/greedy_{args.model}_snngp_lbs")
    plt.close()
    plt.plot(x_axis, ubs)
    plt.savefig(f"figures/greedy_{args.model}_snngp_ubs")
    plt.close()
    plt.plot(x_axis, sharps)
    plt.savefig(f"figures/greedy_{args.model}_snngp_sharps")
    plt.close()
    plt.plot(x_axis, accs)
    plt.savefig(f"figures/greedy_{args.model}_snngp_accs")
    plt.close()
    plt.plot(x_axis, rmses)
    plt.savefig(f"figures/greedy_{args.model}_snngp_rmses")
    plt.close()

if __name__ == "__main__":
    run()
