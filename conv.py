#!/bin/env python

import tensorflow as tf
import numpy as np

from typing import List, Tuple

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
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, layer_depths, hidden_layer_width, 
                 hidden_activation=tf.identity, output_activation=tf.identity):
        
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_depths = layer_depths

        # breakpoint()

        self.inputLayer = Linear(
            num_inputs = self.layer_depths * 100,
            num_outputs = hidden_layer_width
        )
        
        self.hiddenLayer = [
            Linear(num_inputs = hidden_layer_width,
                    num_outputs = hidden_layer_width)
            for i in range(num_hidden_layers)
            ]

        self.outputLayer = Linear(
            num_inputs = hidden_layer_width,
            num_outputs = 10
        )

    def __call__(self,x):
        
        z = self.inputLayer(x)
        
        # print(x.shape)
        for i in range(num_hidden_layers):
            z = self.hidden_activation(self.hiddenLayer[i](z))
            z = tf.nn.dropout(z, 0.5)
        
        # print(x.shape)
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
    def __init__(self, num_inputs, num_outputs, input_depth: int, layer_depths: List[int],
                 layer_kernel_sizes: List[Tuple[int, int]], num_classes: int,
                  num_hidden_layers, hidden_layer_width, 
                  hidden_activation= tf.identity, output_activation=tf.identity):

        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))  ##

        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        

        self.infilter = tf.Variable(
            rng.normal(shape=[layer_kernel_sizes, layer_kernel_sizes, input_depth, layer_depths], stddev=stddev),
            trainable=True,
            name="conv/in",
        )


        self.hidfilter = tf.Variable(
            rng.normal(shape=[layer_kernel_sizes, layer_kernel_sizes, layer_depths, layer_depths], stddev=stddev),
            trainable=True,
            name="conv/hid",
        )

        self.inconv2d = Conv2d(self.infilter, [1,1,1,1])


        self.hidconv2d = Conv2d(self.hidfilter, [1,1,1,1])

        self.fullLayer = MLP(num_inputs, num_outputs, self.num_hidden_layers, layer_depths,
                             hidden_layer_width, self.hidden_activation, self.output_activation)


    def flatten(self, x):
        num_features = x.shape[1] * x.shape[2] * x.shape[3]
        output = tf.reshape(x, [-1, num_features])
        return output

                             
    def __call__(self, x):
        x = self.inconv2d(x)
        # print(x.shape)

        for i in range(8): ## num conv layers
            # x = tf.nn.dropout(tf.nn.relu(self.hidconv2d(x)), .5)
            x = self.hidden_activation(self.hidconv2d(x))
            # print(x.shape)

        x = self.flatten(x)
        
        # print(x.shape)

        x = self.fullLayer(x)
        
        # x = tf.nn.softmax(x)
        # breakpoint()
        # print(x.shape)

        return x

class Adam: # source: https://www.tensorflow.org/guide/core/mlp_core

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
      # Initialize optimizer parameters and variable slots
      self.beta_1 = beta_1
      self.beta_2 = beta_2
      self.learning_rate = learning_rate
      self.ep = ep
      self.t = 1.
      self.v_dvar, self.s_dvar = [], []
      self.built = False

    def apply_gradients(self, grads, vars):
      # Initialize variables on the first call
      if not self.built:
        for var in vars:
          v = tf.Variable(tf.zeros(shape=var.shape))
          s = tf.Variable(tf.zeros(shape=var.shape))
          self.v_dvar.append(v)
          self.s_dvar.append(s)
        self.built = True
      # Update the model variables given their gradients
      for i, (d_var, var) in enumerate(zip(grads, vars)):
        self.v_dvar[i].assign(self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var)
        self.s_dvar[i].assign(self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var))
        v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
        s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
        var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
      self.t += 1.
      return


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



    def oneHotEncode(label):
        onehot = list()
        for value in label:
            row = np.zeros((10,))
            # breakpoint()
            row[value] = 1.0
            label = onehot.append(row)
            # print(label)
        label = tf.cast(onehot, dtype = tf.float32)
        # breakpoint()
        return label

    trainingImages, trainLabels = loadMNIST( "train")
    # testImages, testLabels = loadMNIST( "t10k")

    trainImages = tf.expand_dims(trainingImages / 255.0, -1) # normalize grayscale to 0-1
    trainImages = tf.cast(trainingImages, dtype=tf.float32)

    trainingImages = tf.expand_dims(trainImages[0:40000] / 255.0, -1)
    validImages = tf.expand_dims(trainImages[40001:60000] / 255.0, -1)

    trainingLabels = trainLabels[0:40000]
    validLabels = trainLabels[40001:60000]
    
    trainingLabels = oneHotEncode(trainingLabels)
    validLabels = oneHotEncode(validLabels)
    

    # testImages = tf.expand_dims(testImages / 255.0, -1)

    num_inputs = 1
    num_outputs = 1
    input_layer = 1
    num_classes = 10
    layer_depths = config["conv"]["layer_depths"]
    layer_kernel_sizes = config["conv"]["layer_kernel_sizes"]

    # print(trainingImages.shape)

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

    optimizer = Adam()
    
    ## DISPLAY TRAINING IMAGES ##
    # first_image = np.array(trainingImages[0], dtype='float')
    # pixels = first_image.reshape((28, 28))
    # plt.imshow(pixels, cmap=plt.cm.binary)
    # plt.show()
    #############################


    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)



    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(trainingImages, batch_indices)
            y_batch = tf.gather(trainingLabels, batch_indices)
            # x_batch = tf.gather(validImages, batch_indices)
            # y_batch = tf.gather(validLabels, batch_indices)
            y_batch = tf.cast(y_batch, dtype=tf.float32)
            x_batch = tf.cast(x_batch, dtype=tf.float32)

            y_hat = classifier(x_batch)

            loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_batch, logits = y_hat))


        grads = tape.gradient(loss, classifier.trainable_variables) 
        

        # print(y_hat[0], y_batch[0])
        print(grads[0])
        # breakpoint()

        optimizer.apply_gradients(grads, classifier.trainable_variables)
        # grad_update(step_size, classifier.trainable_variables, grads)


        prediction = tf.math.argmax(y_hat, axis=0)
        prediction = tf.cast(prediction, dtype = tf.float32)
        equality = tf.math.equal(prediction, y_batch)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, Accuracy => {accuracy:.0%}, step_size => {step_size:0.4f}"
            )
            bar.refresh()
