# Data Augmentation in Infinitely Wide Neural Networks

## Pre-requisite
- [JAX](https://github.com/google/jax#installation)

After installing above packages, run following command
```
pip install -r requirements.txt
```

## Usage
To use the SNNGP
```
python snngp_inference.py --model fcn --dataset mnist10k \
--select-method random --num-inducing-points 1000
```

To use the iSNNGP
```
python isnngp_inference.py --model fcn --dataset mnist10k \
--select-method random --num-inducing-points 500 \
--augment-X <path to augmented patterns> --augment-y <path to augmented labels>
```
Note that the augmentations should be stored as .npy files in the 28 x 28 patterns and raw labels.