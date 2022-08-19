#!/bin/bash

python isnngp_inference.py --model cnn \
--batch-size 1000 --device-count -1 --num-inducing-points 50 \
--augment-X /vol/bitbucket/yn621/data/infimnist/mnist120k-augs-patterns-idx3-ubyte --augment-y /vol/bitbucket/yn621/data/infimnist/mnist120k-augs-labels-idx1-ubyte
