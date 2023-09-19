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

class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, 
                 hidden_activation=tf.identity, output_activation=tf.identity):
        
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.num_hidden_layers = num_hidden_layers
    
        self.hidden_activation = hidden_activation

        self.output_activation = output_activation

        self.inputLayer = Linear(
            num_inputs = 1,
            num_outputs = hidden_layer_width
        )
        
        self.hiddenLayer = [
            Linear(num_inputs = hidden_layer_width,
                    num_outputs = hidden_layer_width)
            for i in range(num_hidden_layers)
            ]

        self.outputLayer = Linear(
            num_inputs = hidden_layer_width,
            num_outputs = 1
        )

    def __call__(self,x):
        
        z = self.inputLayer(x)
        
        print(x.shape)
        for i in range(num_hidden_layers):
            z = self.hidden_activation(self.hiddenLayer[i](z))
        
        print(x.shape)
        z = self.output_activation(self.outputLayer(z))
        # print(x.shape)

        return z


class Conv2d(tf.Module):
    def __init__(self, filt, strides):
        self.filt = filt
        self.strides = strides

    def __call__(self, x):
        # breakpoint()
        f = tf.nn.conv2d(x, self.filt, self.strides, padding = 'VALID')
        return f

class Classifier(tf.Module):
    def __init__(self, num_inputs, num_outputs, input_depth: int, layer_depths: list[int],
                 layer_kernel_sizes: list[tuple[int, int]], num_classes: int,
                  num_hidden_layers, hidden_layer_width, 
                  hidden_activation= tf.identity, output_activation=tf.identity):

        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))  ##

        self.num_hidden_layers = num_hidden_layers
        
        self.hidden_activation = hidden_activation

        self.output_activation = output_activation
        
        # self.input_depth = input_depth
        # self.layer_depths = layer_depths
        # self.layer_kernel_sizes = layer_kernel_sizes
        # self.num_classes = num_classes

        # self.filter1 = tf.Variable([[[3.]], [[3.]], [[self.input_depth]], layer_deptgh[0]], dtype=tf.float32)
        
        # self.filter = tf.Variable(tf.random_normal([3,3,5,1]))

        # breakpoint()

        self.infilter = tf.Variable(
            rng.normal(shape=(layer_kernel_sizes, layer_kernel_sizes, input_depth, layer_depths)),
            trainable=True,
            name="conv/f"
        )


        self.hidfilter = tf.Variable(
            rng.normal(shape=(layer_kernel_sizes, layer_kernel_sizes, layer_depths, layer_depths)),
            trainable=True,
            name="conv/f"
        )

        self.inconv2d = Conv2d(self.infilter, [1,1,1,1])


        self.hidconv2d = Conv2d(self.hidfilter, [1,1,1,1])

        # self.flatten = Linear(layer_depths, num_outputs)

        self.fullLayer = MLP(num_inputs, num_outputs, self.num_hidden_layers, 
                             hidden_layer_width, self.hidden_activation, self.output_activation)
                             
    def __call__(self, x):
        x = self.inconv2d(x)
        print(x.shape)

        for i in range(8): ## num conv layers
            x = tf.nn.relu(self.hidconv2d(x))
            print(x.shape)

        x = self.flatten(x)

        print(x.shape)

        breakpoint()

        x = self.fullLayer(x)

        print(x.shape)
        breakpoint()
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

    trainImages = tf.expand_dims(trainingImages / 255.0, -1) # normalize grayscale to 0-1
    trainImages = tf.cast(trainingImages, dtype=tf.float32)

    trainingImages = tf.expand_dims(trainImages[0:40000] / 255.0, -1)
    validImages = tf.expand_dims(trainImages[40001:60000] / 255.0, -1)

    num_inputs = 1
    num_outputs = 1
    input_layer = 1
    num_classes = 10
    layer_depths = config["conv"]["layer_depths"]
    layer_kernel_sizes = config["conv"]["layer_kernel_sizes"]

    print(trainingImages.shape)

    num_samples = config["data"]["num_samples"]
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    classifier = Classifier(num_inputs, num_outputs, input_layer, 
                            layer_depths, layer_kernel_sizes, num_classes,
                            num_hidden_layers, hidden_layer_width, tf.nn.relu, tf.nn.softmax)

    ## DISPLAY TRAINING IMAGES ##
    # first_image = np.array(trainingImages[0], dtype='float')
    # pixels = first_image.reshape((28, 28))
    # plt.imshow(pixels, cmap=plt.cm.binary)
    # plt.show()
    #############################


    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)


#     x = rng.uniform(shape=(num_samples, num_inputs))
#     w = rng.normal(shape=(num_inputs, num_outputs))
#     b = rng.normal(shape=(1, num_outputs))
#     y = rng.normal(
#         shape=(num_samples, num_outputs),
#         mean=x @ w + b,
#         stddev=config["data"]["noise_stddev"],
#     )

#     linear = Linear(num_inputs, num_outputs)

    bar = trange(num_iters)



    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(trainingImages, batch_indices)
            y_batch = tf.reshape(tf.gather(trainingLabels, batch_indices), (batch_size, 1))

            # breakpoint()
            y_hat = classifier(x_batch)

            loss = tf.math.reduce_mean(-y_batch*tf.math.log(y_hat+(1e-7))-(1-y_batch)*tf.math.log(1-y_hat+(1e-7)))


        grads = tape.gradient(loss, conv.trainable_variables) 
        grad_update(step_size, conv.trainable_variables, grads)

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

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
