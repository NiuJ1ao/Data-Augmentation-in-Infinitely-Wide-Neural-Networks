from logger import init_logger
import logger as logging
from optimizer import lbfgs
from jax import grad, jit
import jax.numpy as np
from numpy.random import rand
import torch
import torch.optim as optim
logger = init_logger(log_level=logging.DEBUG)

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)

iterations = 5
 
jax_x = np.float64(pt)
 
# derivative of the objective function
derivative = jit(grad(objective))

logger.info(f"Starting point: {jax_x}")
opt_init, opt_update, get_params = lbfgs(history_size=10,
                                            max_iter=4,)
opt_state = opt_init(jax_x)
for i in range(iterations):
    opt_state = opt_update(i, derivative(get_params(opt_state)), opt_state)
    logger.info(f"-----------Step: {i}; Point: {get_params(opt_state)}-----------")
    
print('\n')
# L-BFGS
x_lbfgs = torch.tensor(pt)
x_lbfgs.requires_grad = True

optimizer = optim.LBFGS([x_lbfgs],
                        history_size=10,
                        max_iter=4,)
h_lbfgs = []
print("Starting point:", x_lbfgs)
for i in range(iterations):
    optimizer.zero_grad()
    o = objective(x_lbfgs)
    o.backward()
    optimizer.step(lambda: objective(x_lbfgs))
    h_lbfgs.append(o.item())
    print(f"-----------Step: {i}; Point: {x_lbfgs.tolist()}-----------")