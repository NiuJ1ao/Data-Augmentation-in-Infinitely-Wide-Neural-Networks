# Gaussian Process implementation based on https://gist.github.com/markvdw/d9fb61eb0c441a4acd38d302d8b4dfdf

import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import gpflow
from typing import Dict
# from gpflow.utilities import to_default_float

from data_loader import load_mnist, load_mnist_full
from inducing_points import random_select, first_n
import logger as logging
logger = logging.init_logger(log_level=logging.INFO)
'''
def gp_mnist():
    original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True)
    total_num_data = info.splits["train"].num_examples
    image_shape = info.features["image"].shape
    image_size = tf.reduce_prod(image_shape)
    batch_size = 60_000


    def map_fn(input_slice: Dict[str, tf.Tensor]):
        updated = input_slice
        image = to_default_float(updated["image"]) / 255.0
        label = to_default_float(updated["label"])
        return tf.reshape(image, [-1, image_size]), label


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
    images_subset, _ = next(iter(dataset))
    images_subset = tf.reshape(images_subset, [-1, image_size])

    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.MultiClass(num_mnist_classes)
    Z = images_subset.numpy()[:num_inducing_points, :]

    model = gpflow.models.SVGP(
        kernel,
        likelihood,
        inducing_variable=Z,
        num_data=total_num_data,
        num_latent_gps=num_mnist_classes,
        whiten=False,
        q_diag=True
    )

    data_iterator = iter(dataset)

    training_loss = model.training_loss_closure(data_iterator)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, variables=model.trainable_variables, options={"maxiter": 5000, "disp": True})

    print(model.elbo)

    original_test_dataset, info = tfds.load(name="mnist", split=tfds.Split.TEST, with_info=True)
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
'''
# Achieves 98.12% accuracy
      
def sgpr_mnist(num_inducing_points=1500):
    train, test = load_mnist(shuffle=False, flatten=True, one_hot=True, val_size=0)

    num_mnist_classes = 10
    num_inducing_points = num_inducing_points  # Can also achieve this with 750 / 1500
    Z, _ = random_select(train[0], num_inducing_points)
    
    train = (tf.convert_to_tensor(train[0]), tf.convert_to_tensor(train[1]))
    test = (tf.convert_to_tensor(test[0]), tf.convert_to_tensor(test[1]))
    Z = tf.convert_to_tensor(Z)
    
    kernel = gpflow.kernels.SquaredExponential()
    
    model = gpflow.models.SGPR(
        data=train,
        kernel=kernel,
        inducing_variable=Z,
        num_latent_gps=num_mnist_classes
    )    

    test_images, test_labels = test
    test_labels = np.argmax(test_labels, axis=-1)
    
    # m, v = model.predict_y(test_images)
    # preds = np.argmax(m, 1).reshape(test_labels.shape)
    # correct = preds == test_labels.astype(int)
    # acc = np.average(correct.astype(float)) * 100.0

    # print("Accuracy: {:.4f}%".format(acc))
    # print(f"ELBO: {model.elbo()}")
    # print(f"EUBO: {model.upper_bound()}")
    
    training_loss = model.training_loss_closure()

    opt = gpflow.optimizers.Scipy()
    res = opt.minimize(training_loss, variables=model.trainable_variables, options={"maxiter": 500, "disp": False})
    logger.info(res)

    m, v = model.predict_y(test_images)
    preds = np.argmax(m, 1).reshape(test_labels.shape)
    correct = preds == test_labels.astype(int)
    acc = np.average(correct.astype(float)) * 100.0

    logger.info("Accuracy: {:.4f}%".format(acc))
    logger.info(f"ELBO: {model.elbo()}")
    logger.info(f"EUBO: {model.upper_bound()}")
  
if __name__ == '__main__':
    for m in np.arange(1000, 11000, 1000):
        logger.info(f"\nNum of inducing points: {m}")
        sgpr_mnist(m)
