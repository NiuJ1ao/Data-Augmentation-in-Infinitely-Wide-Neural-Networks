#!/bin/bash

python snngp_inference.py --model fcn \
--batch-size 0 --device-count -1 --num-inducing-points 10000
