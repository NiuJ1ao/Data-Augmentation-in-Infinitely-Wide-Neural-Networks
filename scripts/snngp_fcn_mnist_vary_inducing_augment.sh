#!/bin/bash

method="random"
model="fcn"

nohup python snngp_vary_inducing_$method.py --model $model --batch-size 0 --device-count -1 \
--select-method $method \
--augment-X /vol/bitbucket/yn621/data/mnist60k-augs-patterns.npy --augment-y /vol/bitbucket/yn621/data/mnist60k-augs-labels.npy \
> log/vary_inducing_${method}_${model}_mnist_aug60k.log 2>&1 &

tail -f log/vary_inducing_${method}_${model}_mnist_aug60k.log