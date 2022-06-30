import numpy as np

def random_select(x, n):
    return x[np.random.choice(x.shape[0], n, replace=False)]


# TODO: greedy variance selection https://github.com/markvdw/RobustGP/blame/0819bc9370f8e974f7f751143224d59d990e9531/robustgp/init_methods/methods.py#L107