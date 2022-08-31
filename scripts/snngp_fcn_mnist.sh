#!/bin/bash

python snngp_inference.py --model fcn \
--batch-size 0 --device-count -1 \
--select-method random --num-inducing-points 1000
