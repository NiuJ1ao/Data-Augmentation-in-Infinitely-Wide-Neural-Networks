#!/bin/bash

method="random"
model="fcn"

nohup python snngp_vary_inducing.py --model fcn --batch-size 0 --device-count -1 --select-method $method > log/vary_inducing_${method}_${model}_mnist.log 2>&1 &
tail -f log/vary_inducing_${method}_${model}_mnist.log