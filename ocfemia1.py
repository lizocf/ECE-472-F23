#!/bin/env python

import tensorflow as tf
import numpy as np


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, M, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable( 
            rng.normal(shape=[M, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        ) # change shape to [M, out] to dot product with X (with shape [in, M])

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

class BasisExpansion(tf.Module):
    def __init__(self, M):

        self.M = M

        self.m = tf.Variable( 
            tf.linspace(0., 1., M), # could have made it in rng distributions?
            trainable=True,
            name="BasisExpansion/m",
        )

        self.s = tf.Variable( 
            0.2*tf.ones(M),
            trainable=True,
            name="BasisExpansion/s",
        ) ##0.2 led to the best prediction

    def __call__(self, x):
        z = tf.exp( -tf.square(x - self.m) / tf.square(self.s))
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

    x = rng.uniform(shape=(num_samples, num_inputs))
    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))
    er = rng.normal(
        shape=(num_samples, num_outputs),
        mean=0,
        stddev=config["data"]["noise_stddev"])
    y = tf.math.sin(2*np.pi*x) + er # random sine samples

    linear = Linear(num_inputs, num_outputs, M) ## add M input for w shape
    basexp = BasisExpansion(M)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            phi = basexp(x_batch)
            y_hat = linear(phi) # integrate base expansion with linear module

            loss = tf.math.reduce_mean(0.5* (y_batch - y_hat) ** 2) # multiply original by 0.5

        grads = tape.gradient(loss, linear.trainable_variables + basexp.trainable_variables) # add all trainable vars for SGD
        grad_update(step_size, linear.trainable_variables + basexp.trainable_variables, grads)

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()


    fig1, ax1 = plt.subplots()

    ax1.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")
    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax1.plot(a.numpy().squeeze(), np.sin(2*np.pi*a), "-")
    ax1.plot(a.numpy().squeeze(), linear(basexp(a)).numpy().squeeze(), ".")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Linear Fit of a Noisy Sinewave using Gaussian Basis Functions")
    
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    fig1.savefig("sine.pdf")

    # Gauss Plots 

    fig2, ax2 = plt.subplots()

    ax2.plot(a.numpy().squeeze(), basexp(a).numpy().squeeze(), "-")

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Gaussian Bases")
    
    h = ax2.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    fig2.savefig("bases.pdf")
    