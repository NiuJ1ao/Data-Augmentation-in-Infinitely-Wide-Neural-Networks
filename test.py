import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from typing import Dict, Optional, Tuple
from gpflow.utilities import to_default_float, set_trainable


original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True, data_dir="data")
total_num_data = info.splits["train"].num_examples
image_shape = info.features["image"].shape
image_size = tf.reduce_prod(image_shape)
batch_size = 60_000


def map_fn(input_slice: Dict[str, tf.Tensor]):
    updated = input_slice
    image = to_default_float(updated["image"]) / 255.0
    label = to_default_float(updated["label"])
    return tf.reshape(image, [-1, image_size]), tf.reshape(label, [-1, 1])


autotune = tf.data.experimental.AUTOTUNE
dataset = (
    original_dataset
    .batch(batch_size, drop_remainder=False)
    .map(map_fn, num_parallel_calls=autotune)
    .prefetch(autotune)
    .repeat()
)

num_mnist_classes = 10
num_inducing_points = 1500  # Can also achieve this with 750
images_subset, y = next(iter(dataset))
images_subset = tf.reshape(images_subset, [-1, image_size])

kernel = gpflow.kernels.SquaredExponential()
likelihood = gpflow.likelihoods.MultiClass(num_mnist_classes)
Z = images_subset.numpy()[:num_inducing_points, :]

model = gpflow.models.SGPR(
    data=next(iter(dataset)),
    kernel=kernel,
    inducing_variable=Z,
    num_latent_gps=num_mnist_classes,
)

data_iterator = iter(dataset)

training_loss = model.training_loss_closure()
opt = gpflow.optimizers.Scipy()
opt.minimize(training_loss, variables=model.trainable_variables, options={"maxiter": 100, "disp": True})



original_test_dataset, info = tfds.load(name="mnist", split=tfds.Split.TEST, with_info=True, data_dir="data")
test_dataset = (
    original_test_dataset
    .batch(10_000, drop_remainder=True)
    .map(map_fn, num_parallel_calls=autotune)
    .prefetch(autotune)
)
test_images, test_labels = next(iter(test_dataset))

m, v = model.predict_y(test_images)
preds = np.argmax(m, 1).reshape(test_labels.numpy().shape)
correct = preds == test_labels.numpy().astype(int)
acc = np.average(correct.astype(float)) * 100.0

print("Accuracy is {:.4f}%".format(acc))

# Achieves 98.12% accuracy