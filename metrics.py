import jax.numpy as np

def mse_predict(mean, var, ys):
  assert mean.shape == var.shape
  assert mean.shape[1] == ys.shape[1]
  
  mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2,
                                   axis=1)
  return mean_predictions # (ts,)

def mse_loss(model):
  return lambda params, x, y: 0.5 * np.mean(np.sum((model.apply_fn(params, x) - y) ** 2, axis=-1))

# def nll(model):
#   return lambda params, x, y: -np.mean(np.sum(y * model.apply_fn(params, x), axis=1))

def accuracy(predicts, targets):
  return np.mean(np.argmax(predicts, axis=1) == np.argmax(targets, axis=1))

def accuracy1d(predicts, targets):
  return np.mean(predicts == targets)

def RMSE(predicts, targets):
  return np.sqrt(np.mean(np.mean(np.square(predicts - targets), axis=0)))
