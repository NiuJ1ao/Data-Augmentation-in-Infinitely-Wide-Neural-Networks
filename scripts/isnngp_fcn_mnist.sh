#!/bin/bash

python isnngp_inference.py --model fcn \
--batch-size 0 --device-count -1 --num-inducing-points 13000 \
--augment-X /vol/bitbucket/yn621/data/mnist50k-augs-patterns.npy --augment-y /vol/bitbucket/yn621/data/mnist50k-augs-labels.npy
