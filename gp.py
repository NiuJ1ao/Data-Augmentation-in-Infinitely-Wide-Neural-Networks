# Gaussian Process implementation based on https://gist.github.com/markvdw/d9fb61eb0c441a4acd38d302d8b4dfdf

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from typing import Dict
from gpflow.utilities import to_default_float

from data_loader import load_mnist
from inducing_points import random_select
from util import create_checkpoint
from tqdm import tqdm

def gp_mnist():
    original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True, data_dir="data")
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
    Z = images_subset.numpy()[:num_inducing_points, :]

    model = gpflow.models.SGPR(
        data=next(iter(dataset)),
        kernel=kernel,
        inducing_variable=Z,
        num_latent_gps=num_mnist_classes,
    )

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
      
def sgpr_mnist():
    train, _, test = load_mnist(flatten=True, one_hot=False)

    num_mnist_classes = 10
    num_inducing_points = 750  # Can also achieve this with 750
    Z = random_select(train[0], num_inducing_points)
    
    kernel = gpflow.kernels.SquaredExponential()
    
    model = gpflow.models.SGPR(
        data=train,
        kernel=kernel,
        inducing_variable=Z,
        num_latent_gps=num_mnist_classes
    )    

    test_images, test_labels = test
    test_labels = np.argmax(test_labels, axis=-1)
    
    training_loss = model.training_loss_closure()

    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, variables=model.trainable_variables, options={"maxiter": 5000, "disp": True})
    
    _, ckpt_manager = create_checkpoint(model=model, output_dir="model/gp/")

    m, v = model.predict_y(test_images)
    preds = np.argmax(m, 1).reshape(test_labels.shape)
    correct = preds == test_labels.astype(int)
    acc = np.average(correct.astype(float)) * 100.0

    print("Accuracy is {:.4f}%".format(acc))
            
    ckpt_path = ckpt_manager.save()
    print(f"Model is saved at {ckpt_path}")
     
 
def svgp_mnist_adam():
    original_dataset, info = tfds.load(name="mnist", split="train[:10000]", with_info=True, data_dir="data/")
    total_num_data = info.splits["train[:10000]"].num_examples
    image_shape = info.features["image"].shape
    image_size = tf.reduce_prod(image_shape)
    batch_size = 200


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
    num_inducing_points = 750  # Can also achieve this with 750      
    data_iterator = iter(dataset)
    images_subset, _ = next(iter(data_iterator))
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


    original_test_dataset, info = tfds.load(name="mnist", split=tfds.Split.TEST, with_info=True, data_dir="data/")
    test_dataset = (
        original_test_dataset
        .batch(1_000, drop_remainder=True)
        .map(map_fn, num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    test_images, test_labels = next(iter(test_dataset))
    
    training_loss = model.training_loss_closure(data_iterator)

    logf = []
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
    
    step_var = tf.Variable(0, dtype=tf.int32, trainable=False)
    _, ckpt_manager = create_checkpoint(model=model, output_dir="model/gp/", step=step_var)

    print("Starting optimisation...")
    for step in tqdm(range(1_000_000)):
        optimization_step()
        step_var.assign_add(1)
        if (step + 1) % 1000 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)

            m, v = model.predict_y(test_images)
            preds = np.argmax(m, 1).reshape(test_labels.numpy().shape)
            correct = preds == test_labels.numpy().astype(int)
            acc = np.average(correct.astype(float)) * 100.0

            print("Step {} - ELBO: {:.4e}, Accuracy is {:.4f}%".format(step + 1, elbo, acc))
            
    ckpt_path = ckpt_manager.save()
    print(f"Model is saved at {ckpt_path}")
    
  
if __name__ == '__main__':
    gp_mnist()
    # sgpr_mnist()
