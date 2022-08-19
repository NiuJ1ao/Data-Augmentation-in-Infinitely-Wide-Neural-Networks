#!/bin/bash

method="random"
model="cnn"

nohup python snngp_vary_inducing.py --model $model --batch-size 2000 --device-count -1 \
--select-method $method \
--augment-X /vol/bitbucket/yn621/data/infimnist/mnist60k-augs-patterns-idx3-ubyte --augment-y /vol/bitbucket/yn621/data/infimnist/mnist60k-augs-labels-idx1-ubyte \
> log/vary_inducing_${method}_${model}_mnist_augment.log 2>&1 &

tail -f log/vary_inducing_${method}_${model}_mnist_augment.log