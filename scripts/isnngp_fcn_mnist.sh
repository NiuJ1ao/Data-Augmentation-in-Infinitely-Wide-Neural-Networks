#!/bin/bash

python isnngp_inference.py --model fcn \
--batch-size 0 --device-count -1 --num-inducing-points 500 \
--augment-X /vol/bitbucket/yn621/data/mnist20k-augs-patterns.npy --augment-y /vol/bitbucket/yn621/data/mnist20k-augs-labels.npy
