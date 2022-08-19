#!/bin/bash

python snngp_inference.py --model cnn \
--batch-size 0 --device-count -1 --num-inducing-points 4450 \
--augment-X /vol/bitbucket/yn621/data/infimnist/mnist60k-augs-patterns-idx3-ubyte --augment-y /vol/bitbucket/yn621/data/infimnist/mnist60k-augs-labels-idx1-ubyte