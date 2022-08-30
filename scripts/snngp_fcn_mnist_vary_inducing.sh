#!/bin/bash

method="greedy"
model="fcn"

nohup python snngp_vary_inducing_$method.py --model $model --batch-size 0 --device-count -1 --select-method $method > log/vary_inducing_${method}_${model}_mnist_10000-20000.log 2>&1 &
tail -f log/vary_inducing_${method}_${model}_mnist_10000-20000.log

# python snngp_vary_inducing_$method.py --model $model --batch-size 0 --device-count -1 --select-method $method