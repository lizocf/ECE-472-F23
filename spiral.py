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

        self.w = tf.Variable(
            rng.normal(shape=[2, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",)
        
    def __call__(self,x):
        for i in range(num_hidden_layers):
            pred = relu(x @ self.w)
            print(pred)
        return pred

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
    spiral_a = np.array([np.cos(theta)*r, np.sin(theta)*r]).T
    spiral_b = np.array([np.cos(theta)*-r, np.sin(theta)*-r]).T
    x_a = spiral_a + np.random.randn(N,1)
    x_b = spiral_b + np.random.randn(N,1)
    ####################

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    refresh_rate = config["display"]["refresh_rate"]

    def relu(x):
        return max(0,x)
    
    # linear = Linear(num_inputs, num_outputs)
    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, relu, relu)

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            xa_batch = tf.gather(x_a, batch_indices)
            xb_batch = tf.gather(x_b, batch_indices)

            Wx_a = mlp(x_a)
            # Wx_b = mlp(x_b)
            
            
        

            # loss = tf.math.reduce_mean(0.5* (y_batch - y_hat) ** 2) # multiply original by 0.5

        # grads = tape.gradient(loss, linear.trainable_variables + basexp.trainable_variables) # add all trainable vars for SGD
        # grad_update(step_size, linear.trainable_variables + basexp.trainable_variables, grads)

        step_size *= decay_rate 

        # if i % refresh_rate == (refresh_rate - 1):
        #     bar.set_description(
        #         f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
        #     )
        #     bar.refresh()

    # loss = -y*log(q)-(1-y)*log(1-q)


    fig1, ax1 = plt.subplots()

    ax1.plot(x_a[:,0], x_a[:,1], "x")
    ax1.plot(x_b[:,0], x_b[:,1], "x")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Spirals")
    
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    fig1.savefig("spiral.pdf")

