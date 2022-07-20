# Gaussian Process implementation based on https://gist.github.com/markvdw/d9fb61eb0c441a4acd38d302d8b4dfdf

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from typing import Dict
from gpflow.utilities import to_default_float

from data_loader import load_mnist
from inducing_points import random_select, first_n
from util import create_checkpoint
from tqdm import tqdm

def gp_mnist():
    original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True, data_dir="data")
    image_shape = info.features["image"].shape
    image_size = tf.reduce_prod(image_shape)
    batch_size = 20_000
    
    original_test_dataset, info = tfds.load(name="mnist", split=tfds.Split.TEST, with_info=True, data_dir="data")

    def map_fn(input_slice: Dict[str, tf.Tensor]):
        updated = input_slice
        image = to_default_float(updated["image"]) / 255.0
        # label = to_default_float(updated["label"])
        label = to_default_float(tf.one_hot(updated["label"], 10) - 0.1)
        return tf.reshape(image, [-1, image_size]), label # tf.reshape(label, [-1, 1])


    autotune = tf.data.experimental.AUTOTUNE
    dataset = (
        original_dataset
        .batch(batch_size, drop_remainder=False)
        .map(map_fn, num_parallel_calls=autotune)
        .prefetch(autotune)
        .repeat()
    )
    test_dataset = (
        original_test_dataset
        .batch(10_000, drop_remainder=True)
        .map(map_fn, num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    test_images, test_labels = next(iter(test_dataset))

    num_mnist_classes = 10
    num_inducing_points = 1500  # Can also achieve this with 750
    images_subset, y = next(iter(dataset))
    images_subset = tf.reshape(images_subset, [-1, image_size])

    kernel = gpflow.kernels.SquaredExponential()
    Z = images_subset.numpy()[:num_inducing_points, :]

    model = gpflow.models.SGPR(
        data=next(iter(dataset)),
        kernel=kernel,
        inducing_variable=Z,
        num_latent_gps=num_mnist_classes,
    )
    
    m, v = model.predict_y(test_images)
    # preds = np.argmax(m, 1).reshape(test_labels.numpy().shape)
    correct = np.argmax(m, 1) == np.argmax(test_labels, 1)
    acc = np.average(correct.astype(float)) * 100.0

    print("Accuracy is {:.4f}%".format(acc))

    training_loss = model.training_loss_closure()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, variables=model.trainable_variables, options={"maxiter": 100, "disp": True})

    m, v = model.predict_y(test_images)
    # preds = np.argmax(m, 1).reshape(test_labels.numpy().shape)
    correct = np.argmax(m, 1) == np.argmax(test_labels, 1)
    acc = np.average(correct.astype(float)) * 100.0

    print("Accuracy is {:.4f}%".format(acc))

# Achieves 98.12% accuracy
      
def sgpr_mnist():
    train, _, test = load_mnist(flatten=True, one_hot=True)

    num_mnist_classes = 10
    num_inducing_points = 1500  # Can also achieve this with 750
    Z = first_n(train[0], num_inducing_points)
    
    kernel = gpflow.kernels.SquaredExponential()
    
    model = gpflow.models.SGPR(
        data=train,
        kernel=kernel,
        inducing_variable=Z,
        num_latent_gps=num_mnist_classes
    )    

    test_images, test_labels = test
    test_labels = np.argmax(test_labels, axis=-1)
    
    m, v = model.predict_y(test_images)
    preds = np.argmax(m, 1).reshape(test_labels.shape)
    correct = preds == test_labels.astype(int)
    acc = np.average(correct.astype(float)) * 100.0

    print("Accuracy is {:.4f}%".format(acc))
    
    training_loss = model.training_loss_closure()

    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, variables=model.trainable_variables, options={"maxiter": 5000, "disp": True})
    
    # _, ckpt_manager = create_checkpoint(model=model, output_dir="model/gp/")

    m, v = model.predict_y(test_images)
    preds = np.argmax(m, 1).reshape(test_labels.shape)
    correct = preds == test_labels.astype(int)
    acc = np.average(correct.astype(float)) * 100.0

    print("Accuracy is {:.4f}%".format(acc))
            
    # ckpt_path = ckpt_manager.save()
    # print(f"Model is saved at {ckpt_path}")
  
if __name__ == '__main__':
    # gp_mnist()
    sgpr_mnist()
