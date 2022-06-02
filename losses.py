import jax.numpy as np

def mse_predict(mean, var, ys):
  assert mean.shape == var.shape
  assert mean.shape[1] == ys.shape[1]
  
  mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2,
                                   axis=1)
  return mean_predictions # (ts,)

def mse_loss(model):
  return lambda params, x, y: 0.5 * np.mean((model.apply_fn(params, x) - y) ** 2)