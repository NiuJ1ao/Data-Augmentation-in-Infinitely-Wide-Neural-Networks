# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.
The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""

import time
import itertools

import numpy.random as npr

import jax.numpy as np
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax

from data_loader import load_mnist
from models import FCN
from nngp import NNGP
from neural_tangents import stax as nt_stax

# model = FCN(num_layers=2, hid_dim=1024, out_dim=10, nonlinearity=nt_stax.Relu)
# loss = lambda params, x, y: 0.5 * np.mean(np.sum((model.apply_fn(params, x) - y) ** 2, axis=-1))
# accuracy = lambda params, x, y: np.mean(np.argmax(model.apply_fn(params, x), axis=1) == np.argmax(y, axis=1))

# init_fn, apply_fn = stax.serial(
#     stax.Dense(1024), stax.Relu,
#     stax.Dense(1024), stax.Relu,
#     stax.Dense(10))

init_fn, apply_fn, _ = nt_stax.serial(
    nt_stax.Dense(1024), nt_stax.Relu(),
    nt_stax.Dense(1024), nt_stax.Relu(),
    nt_stax.Dense(10))

loss = lambda params, x, y: 0.5 * np.mean(np.sum((apply_fn(params, x) - y) ** 2, axis=-1))
accuracy = lambda params, x, y: np.mean(np.argmax(apply_fn(params, x), axis=1) == np.argmax(y, axis=1))

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  # step_size = 0.001
  step_size = 1
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

  (train_images, train_labels), _, (test_images, test_labels) = load_mnist(flatten=True)
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, *batch), opt_state)

  _, init_params = init_fn(rng, train_images.shape)
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  print("\nStarting training...")
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc = accuracy(params, train_images, train_labels).item()
    test_acc = accuracy(params, test_images, test_labels).item()
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}")
  
  