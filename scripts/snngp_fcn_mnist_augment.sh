#!/bin/bash

python snngp_inference.py --model fcn \
--batch-size 0 --device-count -1 --num-inducing-points 5000 \
--augment-X /vol/bitbucket/yn621/data/mnist60k-augs-patterns.npy --augment-y /vol/bitbucket/yn621/data/mnist60k-augs-labels.npy