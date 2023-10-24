#!/bin/env python

import tensorflow as tf
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

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

        # self.w_in = tf.Variable(
        #     rng.normal(shape=[2, hidden_layer_width], stddev=stddev),
        #     trainable=True,
        #     name="Linear/w_in",
        # )
        # self.w_h = tf.Variable(
        #     rng.normal(shape=[hidden_layer_width, hidden_layer_width, num_hidden_layers], stddev=stddev),
        #     trainable=True,
        #     name="Linear/w_h",
        # )

        # self.w_out = tf.Variable(
        #     rng.normal(shape=[hidden_layer_width, 1], stddev=stddev),
        #     trainable=True,
        #     name="Linear/w_out",
        # )

        self.num_hidden_layers = num_hidden_layers
    

        self.inputLayer = Linear(
            num_inputs = 2,
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

        self.hidden_activation = hidden_activation

        self.output_activation = output_activation


    def __call__(self,x):

        z = self.inputLayer(x)
        
        # print(x.shape)
        for i in range(num_hidden_layers):
            z = self.hidden_activation(self.hiddenLayer[i](z))
        
        # print(x.shape)
        z = self.output_activation(self.outputLayer(z))
        print(x.shape)

        return z

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)
 
if __name__ == "__main__":
    import argparse
    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange
    import numpy as np

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    num_inputs = 1
    num_outputs = 1

    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))

    
    #### SPIRAL GEN ####
    N = np.int32(num_samples / 2)
    theta = np.sqrt(np.random.rand(N))*3*np.pi

    r = 2*theta + np.pi
    spiral_a = np.array([tf.math.cos(theta)*r, tf.math.sin(theta)*r]).T
    spiral_b = np.array([tf.math.cos(theta)*-r, tf.math.sin(theta)*-r]).T
    x_a = spiral_a + np.random.rand(N,1)
    x_b = spiral_b + np.random.rand(N,1)

    x = tf.Variable(np.append(x_a, x_b, axis=0))
    x = tf.cast(x, dtype=tf.float32)

    y = tf.Variable(np.append(tf.zeros(x_a.shape[0]), tf.ones(x_b.shape[0]), axis=0))
    ####################

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    refresh_rate = config["display"]["refresh_rate"]

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, tf.nn.relu, tf.nn.sigmoid)

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.reshape(tf.gather(y, batch_indices), (batch_size,1)) 
            ## THIS TOOK ME HOURS TO REALIZE, grad kept outputting 0, turns out y_batch was shaped wrong. ty Gary

            y_hat = mlp(x_batch)

            loss = tf.math.reduce_mean(-y_batch*tf.math.log(y_hat+(1e-7))-(1-y_batch)*tf.math.log(1-y_hat+(1e-7)))

        grads = tape.gradient(loss, mlp.trainable_variables)

        
        # breakpoint()
        grad_update(step_size, mlp.trainable_variables, grads)

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, step_size => {step_size}"
            )
            bar.refresh()

    fig1, ax1 = plt.subplots()

    ax1.plot(x_a[:,0], x_a[:,1], "x")
    ax1.plot(x_b[:,0], x_b[:,1], "x")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Spirals")
    
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)
    fig1.savefig("dataset.pdf")


    f1, f2 = np.meshgrid(np.linspace(-25.,25., 600), np.linspace(-25.,25.,600))
    grid = np.vstack([f1.ravel(), f2.ravel()]).T
    y_pred = np.reshape(mlp(grid), f1.shape)
    display = DecisionBoundaryDisplay(xx0=f1, xx1=f2, response=y_pred)
    display.plot()

    display.ax_.scatter(x_a[:,0], x_a[:,1])
    display.ax_.scatter(x_b[:,0], x_b[:,1])

    plt.savefig("spiral.pdf")