# Imagine A Deep Network That Performs Successive Cheap Queries On Its Input

Deep networks typically perform a single parallel function on the input. How would they be different if they performed an iterative sequence of smaller functions?

See blog post: [Imagine A Deep Network That Performs Successive Cheap Queries On Its Input](https://probablymarcus.com/blocks/2022/07/08/deep-adaptive-successive-queries.html)

In progress, much of the code is not yet polished for others' eyes. Some of the code is poorly documented, there are multiple versions of it. The hyperparameters in the examples might not be the best ones I've found. Everything is subject to change. This repo contains working code for both PyTorch and TensorFlow. There is not feature parity between the two; the TensorFlow code is much more up-to-date. (I started this project in PyTorch then moved to TensorFlow so that I could take advantage of my M1 MacBook Pro.)

The most up-to-date model code is [subsequent/tensorflow/conv2d_model.py](subsequent/tensorflow/conv2d_model.py), which can be run via [examples/tensorflow/run_mnist.py](examples/tensorflow/run_mnist.py) and [examples/tensorflow/run_cifar.py](examples/tensorflow/run_cifar.py).
