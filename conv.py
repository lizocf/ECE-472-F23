#!/bin/env python

import tensorflow as tf
import numpy as np

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))  ##

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z

class Conv2d(tf.Module):
    def __init__(self, filt, strides):
        self.filt = filt
        self.strides = strides

    def __call__(self, x):
        f = tf.nn.conv2d(x, self.filt, self.strides, padding = 'VALID')
        return f

class Classifier(tf.Module):
    def __init__(self, input_depth: int, layer_depths: list[int],
                 layer_kernel_sizes: list[tuple[int, int]], num_classes: int):
        
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes

        self.filter = tf.Variable([self.layer_kernel_sizes, 1, self.input_depth])

        self.conv2d = Conv2d(self.filter, [1,1,1,1])

    def __call__(self, x):
        for i in range(self.input_depth):
            x = self.conv2d(x)
        print(x.shape)
        return x


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    ## silly little data preprocessing ##
    def loadMNIST(prefix): # was not quite sure how to implement loading without mnist, ended up using stackoverflow solution
        intType = np.dtype( 'int32' ).newbyteorder( '>' )
        nMetaDataBytes = 4 * intType.itemsize

        data = np.fromfile(prefix + '-images.idx3-ubyte', dtype = 'ubyte' )
        magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
        data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

        labels = np.fromfile(prefix + '-labels.idx1-ubyte', dtype = 'ubyte' )[2 * intType.itemsize:]
        return data, labels

    trainingImages, trainingLabels = loadMNIST( "train")
    # testImages, testLabels = loadMNIST( "t10k")

    trainingImages = trainingImages / 255.0 # normalize grayscale to 0-1
    print(trainingImages.shape)

    
    classifier = Classifier(64, list[32, 16, 10], 3, 10)
    y_hat = classifier([trainingImages, 1])

    ## DISPLAY TRAINING IMAGES ##
    # first_image = np.array(trainingImages[0], dtype='float')
    # pixels = first_image.reshape((28, 28))
    # plt.imshow(pixels, cmap=plt.cm.binary)
    # plt.show()
    #############################


#     rng = tf.random.get_global_generator()
#     rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

#     num_samples = config["data"]["num_samples"]
#     num_inputs = 1
#     num_outputs = 1

#     x = rng.uniform(shape=(num_samples, num_inputs))
#     w = rng.normal(shape=(num_inputs, num_outputs))
#     b = rng.normal(shape=(1, num_outputs))
#     y = rng.normal(
#         shape=(num_samples, num_outputs),
#         mean=x @ w + b,
#         stddev=config["data"]["noise_stddev"],
#     )

#     linear = Linear(num_inputs, num_outputs)

#     num_iters = config["learning"]["num_iters"]
#     step_size = config["learning"]["step_size"]
#     decay_rate = config["learning"]["decay_rate"]
#     batch_size = config["learning"]["batch_size"]

#     refresh_rate = config["display"]["refresh_rate"]

#     bar = trange(num_iters)

#     for i in bar:
#         batch_indices = rng.uniform(
#             shape=[batch_size], maxval=num_samples, dtype=tf.int32
#         )
#         with tf.GradientTape() as tape:
#             x_batch = tf.gather(x, batch_indices)
#             y_batch = tf.gather(y, batch_indices)

#             y_hat = linear(x_batch)
#             loss = tf.math.reduce_mean((y_batch - y_hat) ** 2) ##

#         grads = tape.gradient(loss, linear.trainable_variables) 
#         grad_update(step_size, linear.trainable_variables, grads)

#         step_size *= decay_rate 

#         if i % refresh_rate == (refresh_rate - 1):
#             bar.set_description(
#                 f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
#             )
#             bar.refresh()

#     fig, ax = plt.subplots()

#     ax.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")

#     a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
#     ax.plot(a.numpy().squeeze(), linear(a).numpy().squeeze(), "-")

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title("Linear fit using SGD")
    
#     h = ax.set_ylabel("y", labelpad=10)
#     h.set_rotation(0)

#     fig.savefig("plot.pdf")
