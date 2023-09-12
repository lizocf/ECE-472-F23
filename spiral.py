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

        self.w_in = tf.Variable(
            rng.normal(shape=[2, hidden_layer_width], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )
        self.w_h = tf.Variable(
            rng.normal(shape=[hidden_layer_width, hidden_layer_width], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.w_out = tf.Variable(
            rng.normal(shape=[hidden_layer_width, 2], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.num_hidden_layers = num_hidden_layers
        
    def __call__(self,x):
        
        # input layer [bs, n]
        z = x @ self.w_in

        # hidden layer(s) [n, m] -> [m, m]
        for i in range(self.num_hidden_layers):
            z = relu(z @ self.w_h)
            # print(z)
        
        # output layer [m, out]
        # pred = sigmoid(pred @ self.w_out)
        z = z @ self.w_out
        return z

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)

if __name__ == "__main__":
    import argparse

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
    M = 6

    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))

    
    #### SPIRAL GEN ####
    N = 200
    theta = np.sqrt(np.random.rand(N))*3*np.pi

    r = 2*theta + np.pi
    spiral_a = np.array([tf.math.cos(theta)*r, tf.math.sin(theta)*r]).T
    spiral_b = np.array([tf.math.cos(theta)*-r, tf.math.sin(theta)*-r]).T
    x_a = spiral_a + np.random.rand(N,1)
    x_b = spiral_b + np.random.rand(N,1)
    x = tf.Variable(np.append(x_a, x_b, axis=0))
    x = tf.cast(x, dtype=tf.float32)

    y = tf.Variable(np.append(tf.zeros(x_a.shape[0]), tf.ones(x_b.shape[0]), axis=0))
    y = tf.cast(x, dtype=tf.float32)
    ####################

    # print(x[:,1])

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    refresh_rate = config["display"]["refresh_rate"]

    def relu(x):
        # return max(0,x)
        return tf.math.maximum(0,x)

    def sigmoid(x):
        return 1 / (1 + tf.math.exp(-x))
    
    # linear = Linear(num_inputs, num_outputs)
    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)
    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            y_hat = mlp(x_batch)
            print(y_batch)

            loss = tf.math.reduce_mean(-y_batch*tf.math.log(y_hat+(1e-9))-(1-y_batch)*tf.math.log(1-y_hat+(1e-9)))
            # print(loss)            
            # loss = -(y_batch)*tf.math.log(y_hat) - (1-y_batch)*tf.math.log(y_hat)
        grads = tape.gradient(loss, mlp.trainable_variables) # add all trainable vars for SGD
        grad_update(step_size, mlp.trainable_variables, grads)

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, step_size => {step_size}"
            )
            bar.refresh()



    fig1, ax1 = plt.subplots()

    # print(x_a.shape[0])

    ax1.plot(x_a[:,0], x_a[:,1], "x")
    ax1.plot(x_b[:,0], x_b[:,1], "x")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Spirals")
    
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    display = DecisionBoundaryDisplay.from_estimator()

    fig1.savefig("spiral.pdf")

