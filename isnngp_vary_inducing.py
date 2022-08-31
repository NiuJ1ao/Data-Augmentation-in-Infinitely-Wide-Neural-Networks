import jax.numpy as np
import matplotlib.pyplot as plt

import logger as logging
from metrics import RMSE, accuracy
from isnngp import iSNNGP
from isnngp_inference import prepare_model_and_data
from inducing_points import select_inducing_points
from util import args_parser, init_random_state

logger = logging.init_logger(log_level=logging.INFO)
from jax.config import config

config.update("jax_enable_x64", True)

def run():
    args = args_parser()
    init_random_state(0)
    model, model_params, train, augments, test = prepare_model_and_data(args)
    train_x, train_y = train
    test_x, test_y = test
    stds = np.array([1.1, 0.15], dtype=np.float64)
    noise_variance = 0.01
    lbs, ubs, losses, accs = [], [], [], []
    nums_inducing_points = np.arange(50, 5050, 50)
    for m in nums_inducing_points:
        logger.info(f"\n----Number of inducing point: {m}----")
        inducing_points, _ = select_inducing_points(args.select_method, train_x, m, model, stds, model_params)
        
        isnngp = iSNNGP(model=model, model_params=model_params, train_data=train, train_augs=augments, inducing_points=inducing_points, num_latent_gps=10, init_stds=stds, batch_size=args.batch_size, noise_variance=noise_variance)
        
        # isnngp.optimize()

        elbo = isnngp.lower_bound()
        logger.info(f"ELBO: {elbo}")
        eubo = isnngp.upper_bound()
        logger.info(f"EUBO: {eubo}")
        mean, var = isnngp.predict(test_x, diag=True)
        logger.debug(f"mean: {mean.shape}; cov: {var.shape}")
        loss = RMSE(mean, test_y)
        logger.info(f"Loss: {loss}")
        acc = accuracy(mean, test_y)
        logger.info(f"Accuracy: {acc:.2%}")
        
        lbs.append(elbo)
        ubs.append(eubo)
        losses.append(loss)
        accs.append(acc)
    
    plt.plot(nums_inducing_points, lbs)
    plt.xlabel("Num of inducing points")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig(f"figures/isnngp_vary_inducing_{args.select_method}_{args.model}lbs.pdf")
    plt.close()
    plt.plot(nums_inducing_points, ubs)
    plt.xlabel("Num of inducing points")
    plt.ylabel("EUBO")
    plt.legend()
    plt.savefig(f"figures/isnngp_vary_inducing_{args.select_method}_{args.model}ubs.pdf")
    plt.close()
    plt.plot(nums_inducing_points, losses)
    plt.xlabel("Num of inducing points")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"figures/isnngp_vary_inducing_{args.select_method}_{args.model}rmses.pdf")
    plt.close()
    plt.plot(nums_inducing_points, accs)
    plt.xlabel("Num of inducing points")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"figures/isnngp_vary_inducing_{args.select_method}_{args.model}accs.pdf")
    plt.close()
    
if __name__ == "__main__":
    run()
