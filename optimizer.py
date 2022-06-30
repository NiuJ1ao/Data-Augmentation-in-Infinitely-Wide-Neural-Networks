from jax.example_libraries.optimizers import Optimizer, optimizer

@optimizer
def sgd(step_size):
    """Construct optimizer triple for stochastic gradient descent.

    Args:
        step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    def init(x0):
        return x0
    def update(i, g, state):
        x = state
        return x - step_size(i) * g
    def get_params(state):
        return state
    return Optimizer(init, update, get_params)

@optimizer
def dummy_optimizer():
    return 