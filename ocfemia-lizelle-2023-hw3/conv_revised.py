#!/bin/env python

import tensorflow as tf
import numpy as np

from typing import List, Tuple

class Conv2d(tf.Module):
    def __init__(self, layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation):

        rng = tf.random.get_global_generator()
        self.hidden_activation = hidden_activation

        self.infilter = tf.Variable(
            rng.normal(shape=[layer_kernel_sizes, layer_kernel_sizes, input_depth, layer_depths], 
            stddev= tf.math.sqrt(2 / (input_depth + layer_depths))) ,
            trainable=True,
            name="conv/in",
        )
        self.hidfilter = tf.Variable(
            rng.normal(shape=[layer_kernel_sizes, layer_kernel_sizes, layer_depths, layer_depths], 
            stddev= tf.math.sqrt(2 / (layer_depths*2)) ),
            trainable=True,
            name="conv/hid",
        )
        self.ffilter = tf.Variable(
            rng.normal(shape=[1, 1, layer_depths, num_classes], 
            stddev= tf.math.sqrt(2 / (layer_depths + num_classes)) ),
            trainable=True,
            name="conv/f"
        )
        
    def __call__(self, x):
        f = tf.nn.conv2d(x, self.infilter, [1,1,1,1], padding = 'VALID')
        for i in range(8):
            f = self.hidden_activation(tf.nn.conv2d(f, self.hidfilter, [1,1,1,1], padding = 'VALID'))
        print(f.shape)
        f = tf.math.reduce_mean(f, axis = [1,2], keepdims=True)
        print(f.shape)
        f = tf.nn.conv2d(f, self.ffilter, [1,1,1,1], padding = 'VALID')
        return f

class Classifier(tf.Module):
    def __init__(self, input_depth: int, layer_depths: List[int], layer_kernel_sizes: List[Tuple[int, int]], 
                num_classes: int, hidden_activation= tf.identity):
        self.conv2d = Conv2d(layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation)

    def __call__(self, x):
        x = self.conv2d(x)
        return tf.squeeze(x)

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

    ## data preprocessing ##
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

    trainingImages, trainLabels = loadMNIST( "/home/lizocf/ECE-471-DL/ocfemia-lizelle-2023-hw3/datasets/train")
    testImages, testLabels = loadMNIST( "/home/lizocf/ECE-471-DL/ocfemia-lizelle-2023-hw3/datasets/t10k")

    trainImages = tf.expand_dims(trainingImages / 255.0, -1) # normalize grayscale to 0-1
    trainImages = tf.cast(trainingImages, dtype=tf.float32)

    trainingImages = tf.expand_dims(trainImages[0:40000] / 255.0, -1)
    validImages = tf.expand_dims(trainImages[40001:60000] / 255.0, -1)
    testImages = tf.expand_dims(testImages / 255.0, -1)

    trainingLabels = trainLabels[0:40000]
    validLabels = trainLabels[40001:60000]
    
    trainingLabels = oneHotEncode(trainingLabels)
    validLabels = oneHotEncode(validLabels)
    testLabels = oneHotEncode(testLabels)


    input_layer = 1
    num_classes = 10
    layer_depths = config["conv"]["layer_depths"]
    layer_kernel_sizes = config["conv"]["layer_kernel_sizes"]

    num_samples = config["data"]["num_samples"]
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    classifier = Classifier(input_layer, layer_depths, layer_kernel_sizes, num_classes, tf.nn.relu)

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
            # x_batch = tf.gather(trainingImages, batch_indices)
            # y_batch = tf.gather(trainingLabels, batch_indices)

            # x_batch = tf.gather(validImages, batch_indices)
            # y_batch = tf.gather(validLabels, batch_indices)
            
            x_batch = tf.gather(testImages, batch_indices)
            y_batch = tf.gather(testLabels, batch_indices)
            
            y_batch = tf.cast(y_batch, dtype=tf.float32)
            x_batch = tf.cast(x_batch, dtype=tf.float32)

            y_hat = classifier(x_batch)

            loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_batch, logits = y_hat))

        grads = tape.gradient(loss, classifier.trainable_variables) 
        breakpoint()
        
        optimizer.apply_gradients(grads, classifier.trainable_variables)

        prediction = tf.math.argmax(y_hat, axis=-1)
        y_batch = tf.math.argmax(y_batch, axis=-1)
        equality = tf.math.equal(prediction, y_batch)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, Accuracy => {accuracy:.0%}, step_size => {step_size:0.4f}"
            )
            if accuracy >= .955:
                with open('acc_loss_config.txt', 'a') as f:
                    f.write(f"-----STEP_SIZE: {step_size}, BATCH_SIZE: {batch_size}, LAYER_DEPTH: {layer_depths} -----\n")
                    f.write(f"Accuracy: {accuracy:.0%}. Loss: {loss}. Steps Taken: {i}. \n")
                exit()
            bar.refresh()


# TRAINING: 
#       Accuracy: 97%, Steps Taken: 174, Loss: 0.12450236082077026
# VALIDATION: 
#       Using Training config: Accuracy: 96%. Steps Taken: 184, Loss: 0.15900187194347382
#       Best Tuning: 
#           -----STEP_SIZE: 0.08824417114557709, BATCH_SIZE: 250, LAYER_DEPTH: 100 -----
#                   Accuracy: 96%. Steps Taken: 124. Loss: 0.21166913211345673. 
# TEST: 
# -----STEP_SIZE: 0.08478237090774315, BATCH_SIZE: 250, LAYER_DEPTH: 100 -----
#           Accuracy: 97%. Loss: 0.18395516276359558. Steps Taken: 164. 
