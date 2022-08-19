from imgaug import augmenters as iaa
from data_loader import load_mnist
import jax.numpy as jnp
import numpy as np

def image_augmenter(image, orbit_size=1, rotate=(-45, 45), crop_percentage=(0.15, 0.15), image_shape=(28, 28)):
    augs = []
    for _ in range(orbit_size):
        seq = iaa.Sequential([
            iaa.Affine(rotate=rotate),
            iaa.Crop(percent=crop_percentage)
        ])
        aug_image = seq(image=np.array(image).reshape(image_shape))
        augs.append(jnp.expand_dims(aug_image, 0))
        
    return jnp.concatenate(augs)

def images_augmenter(images, orbit_size=1, rotate=(-45, 45), crop_percentage=(0.15, 0.15)):
    augs = []
    for _ in range(orbit_size):
        seq = iaa.Sequential([
            iaa.Affine(rotate=rotate),
            iaa.Crop(percent=crop_percentage)
        ])
        aug_images = seq(images=np.array(images))
        augs.append(jnp.expand_dims(aug_images, 1))
    return jnp.concatenate(augs, 1)

if __name__ == "__main__":
    train, _, _ = load_mnist()
    augs = images_augmenter(train[0], orbit_size=2)
    print(augs.shape)
    # jnp.save("/vol/bitbucket/yn621/data/mnist_aug_X", augs)
    # jnp.save("/vol/bitbucket/yn621/data/mnist_aug_y", train[1])
    # augs = jnp.load("/vol/bitbucket/yn621/data/mnist_aug_X.npy")
    # print(augs.shape)