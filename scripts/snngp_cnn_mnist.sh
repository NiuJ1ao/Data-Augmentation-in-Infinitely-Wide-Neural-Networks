#!/bin/bash

python snngp_inference.py --model cnn \
--batch-size 2000 --device-count -1 --num-inducing-points 5000
