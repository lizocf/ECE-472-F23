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
            rng.normal(shape=[1, 1, layer_depths, input_depth], 
            stddev= tf.math.sqrt(2 / (layer_depths + input_depth)) ),
            trainable=True,
            name="conv/f"
        )
        
    def __call__(self, x, final=False):
        f = tf.nn.conv2d(x, self.infilter, [1,1,1,1], padding = 'SAME')
        for i in range(8):
            f = self.hidden_activation(tf.nn.conv2d(f, self.hidfilter, [1,1,1,1], padding = 'SAME'))
        f = tf.nn.conv2d(f, self.ffilter, [1,1,1,1], padding = 'SAME')
        return f

class Conv1x1(tf.Module):
    def __init__(self, layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation):
        rng = tf.random.get_global_generator()
        self.hidden_activation = hidden_activation

        self.ffilter = tf.Variable(
            rng.normal(shape=[1, 1, input_depth, num_classes], 
            stddev= tf.math.sqrt(2 / (input_depth + num_classes)) ),
            trainable=True,
            name="conv/f"
        )
    def __call__(self, x):
        f = tf.math.reduce_mean(x, axis = [1,2], keepdims=True)
        f = tf.nn.conv2d(f, self.ffilter, [1,1,1,1], padding = 'SAME')
        return f

class GroupNorm(tf.Module): # see Group Normalization by Wu et al. https://arxiv.org/pdf/1803.08494.pdf 
    def __init__(self, G=32, eps=1e-5):
        self.G = G
        self.eps = eps

        self.gamma = tf.Variable(
            tf.ones(shape = [1,3,1,1]), 
            trainable=True, 
            name="gn/gamma")
            
        self.beta = tf.Variable(
            tf.ones(shape = [1,3,1,1]), 
            trainable=True, 
            name="gn/beta")
            
    def __call__(self,x):
        # x: input features with shape [N,C,H,W]
        # gamma, beta: scale and offset, with shape [1,C,1,1]
        # G: number of groups for GN
        x = tf.transpose(x, [0,3,1,2]) # change [bs,h,w,ch] to [bs,ch,h,w] according to paper
        N, C, H, W = x.get_shape().as_list()
        self.G = min(self.G,C)
        x = tf.reshape(x, [-1, self.G, C // self.G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [-1, C, H, W])
        x = x * self.gamma + self.beta
        x = tf.transpose(x, [0,2,3,1]) # revert back to [bs,h,w,ch]
        return x


class ResidualBlock(tf.Module):
    def __init__(self, layer_kernel_sizes,input_depth,layer_depths,num_classes, hidden_activation):
        self.hidden_activation = hidden_activation
        self.conv2d = Conv2d(layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation)
        self.gnorm = GroupNorm(G=32)
    def __call__(self,x):
        f = self.conv2d(x) # [bs,32,32,3]
        f = self.gnorm(f) # [bs,32,32,3]
        f = tf.nn.dropout(f,0.5)
        x = f + x
        return x
        

class Classifier(tf.Module):
    def __init__(self, input_depth: int, layer_depths: List[int], layer_kernel_sizes: List[Tuple[int, int]], 
                num_classes: int, num_res_blocks = List[int], hidden_activation= tf.identity):
        self.num_res_blocks = num_res_blocks
        self.hidden_activation = hidden_activation
        self.resblock = ResidualBlock(layer_kernel_sizes,input_depth,layer_depths,num_classes, hidden_activation)
        self.conv1x1 = Conv1x1(layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation)

    def __call__(self, x):
        for i in range(3):
            x = self.resblock(x)
        x = self.hidden_activation(x)
        x = self.conv1x1(x) # [bs,1,1,10]
        x = tf.squeeze(x) # [bs,10]
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
        # breakpoint()
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
    
    from sklearn.metrics import top_k_accuracy_score

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config10.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def oneHotEncode(label):
        onehot = list()
        for value in label:
            row = np.zeros((10,))
            row[value] = 1.0
            label = onehot.append(row)
        label = tf.cast(onehot, dtype = tf.float32)
        return label


    def getImages(file,valid=False,test=False):
        combined_data = []
        combined_labels = []
        if test:
            batch = unpickle(file)
            data = batch[b'data']
            data = data.reshape(len(data),3,32,32).transpose(0,2,3,1) / 255.0
            labels = batch[b'labels']
            combined_labels.append(labels)
            combined_data.append(data)
            return combined_data[0], combined_labels[0]
        if valid:
            batch = unpickle(file + str(5))
            data = batch[b'data']
            data = data.reshape(len(data),3,32,32).transpose(0,2,3,1) / 255.0
            labels = batch[b'labels']
            combined_labels.append(labels)
            combined_data.append(data)
            return combined_data[0], combined_labels[0]
        else:
            for i in range(4):
                batch = unpickle(file + str(i+1))
                data = batch[b'data']
                data = data.reshape(len(data),3,32,32).transpose(0,2,3,1) / 255.0
                labels = batch[b'labels']
                combined_labels.append(labels)
                combined_data.append(data)
            combined_data = np.concatenate([combined_data[0],combined_data[1],
                                            combined_data[2],combined_data[3]], axis=0)
            combined_labels = np.concatenate([combined_labels[0],combined_labels[1],
                                            combined_labels[2],combined_labels[3]], axis=0)
            return combined_data, combined_labels


    file = r'/workdir/ocfemia-lizelle-2023-hw4/datasets/cifar-10-batches-py/data_batch_' # for docker :)
    # file = '/home/lizocf/ECE-471-DL/ocfemia-lizelle-2023-hw4/datasets/cifar-10-batches-py/data_batch_' # for local :)
    
    trainingImages, trainingLabels = getImages(file)
    trainingLabels = tf.expand_dims(trainingLabels, -1)
    trainingLabels = oneHotEncode(trainingLabels)

    validImages, validLabels = getImages(file,valid=True)
    validLabels = tf.expand_dims(validLabels, -1)
    validLabels = oneHotEncode(validLabels)

    file_test = r'/workdir/cifar-10-batches-py/test_batch'
    testImages, testLabels = getImages(file_test,test=True)
    testLabels = tf.expand_dims(testLabels, -1)
    testLabels = oneHotEncode(testLabels)
    

    input_layer = 3
    num_classes = 10
    layer_depths = config["conv"]["layer_depths"]
    layer_kernel_sizes = config["conv"]["layer_kernel_sizes"]
    num_res_blocks = config["conv"]["num_res_blocks"]

    num_samples = config["data"]["num_samples"]
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    classifier = Classifier(input_layer, layer_depths, layer_kernel_sizes, num_classes, num_res_blocks, tf.nn.relu)

    optimizer = Adam()
    
    iters = []
    accuracies = []
    losses = []

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

            x_batch = tf.image.random_flip_left_right(x_batch)
            
            y_hat = classifier(x_batch)

            loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_batch, logits = y_hat))

        grads = tape.gradient(loss, classifier.trainable_variables) 

        optimizer.apply_gradients(grads, classifier.trainable_variables)

        prediction = tf.math.argmax(y_hat, axis=-1)
        y_batch = tf.math.argmax(y_batch, axis=-1)
        equality = tf.math.equal(prediction, y_batch)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

        step_size *= decay_rate 

        iters.append(i)
        accuracies.append(accuracy)
        losses.append(loss)
        
        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, Accuracy => {accuracy:.0%}, step_size => {step_size:0.4f}"
            )
            if accuracy >= .99:
                with open('acc_loss_config.txt', 'a') as f:
                    f.write(f"-----STEP_SIZE: {step_size}, BATCH_SIZE: {batch_size}, LAYER_DEPTH: {layer_depths} -----\n")
                    f.write(f"Accuracy: {accuracy:.0%}. Loss: {loss}. Steps Taken: {i}. \n")
                print("num_params", tf.math.add_n([tf.math.reduce_prod(var.shape) for var in classifier.trainable_variables]))
                fig, axs = plt.subplots(1, 2)
                axs[0, 0].plot(iters, losses, '-')
                axs[0, 0].set_title('Loss per Iteration')
                axs[0, 1].plot(iters, accuracies, '-')
                axs[0, 1].set_title('Accuracy per Iteration')
                plt.savefig("Cifar10.pdf", format='pdf')
                plt.show()
                exit()
            bar.refresh()

# lzl notes:
#  - MNIST model works incredibly slow with the cifar dataset, res. implementation will speed up this process
#  - input & output layers must be same size for addition -> padding = SAME
#  - data augmentation: try flipping 50%? Unfortunately, I had to flip all input images because of RESOURCE_EXHAUSTED: failed to allocate memory
