#!/bin/bash

if [ ! -d ./data/mnist ]; then
  mkdir -p ./data/mnist;
fi

pushd data/mnist
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
popd