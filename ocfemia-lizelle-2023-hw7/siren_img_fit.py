#!/bin/env python

import tensorflow as tf
import numpy as np

SIZE = 256

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, omega_0, bias=True, is_first=False):
        rng = tf.random.get_global_generator()

        if is_first: # important ! I forgot to initialize weights and kept getting white noise as predictions, made complete sense tho >:(
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], 
                minval=-1/num_inputs, 
                maxval= 1/num_inputs),
                trainable=True,
                name="Linear/fw" 
            )
        else:            
            self.w = tf.Variable( 
                rng.uniform(shape=[num_inputs, num_outputs], 
                minval=-np.sqrt(6 / num_inputs) / omega_0,
                maxval= np.sqrt(6 / num_inputs) / omega_0),
                trainable=True,
                name="Linear/hw",
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

class SineLayer(tf.Module):
    def __init__(self,in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = Linear(in_features, out_features, omega_0, bias=bias, is_first=self.is_first)

    def __call__(self, input):
        return tf.math.sin(self.omega_0 * self.linear(input))

class Siren(tf.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.inputLayer = SineLayer(in_features, hidden_features, 
                                    is_first=True, omega_0=first_omega_0)  

        self.hiddenLayer = [SineLayer(hidden_features, hidden_features, 
                                       is_first=False, omega_0=hidden_omega_0)
                            for i in range(self.hidden_layers)]
        
        self.outputLayer = Linear(hidden_features, out_features, hidden_omega_0)

    def __call__(self, x):
        z = self.inputLayer(x)
        # breakpoint()

        for i in range(self.hidden_layers):
            z = tf.nn.relu(self.hiddenLayer[i](z))
        z = self.outputLayer(z)
        # z = tf.reshape(z, [self.hidden_features,self.hidden_features,-1])
        return z

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
    import numpy as np

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits image",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)


    def get_mgrid(sidelen, dim=2): # stolen from siren colab
        """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int"""
        tensors = tuple(dim * [tf.linspace(-1, 1, num=sidelen)])
        mgrid = tf.stack(tf.meshgrid(*tensors, indexing="ij"), axis=-1)
        mgrid = tf.reshape(mgrid, [-1, dim])
        return mgrid

    def get_img(img_path, img_size):
        img_raw = tf.io.read_file(img_path)
        img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)
        img_ground_truth = tf.image.resize(img_ground_truth, [img_size, img_size])
        return (
            get_mgrid(img_size, 2),
            tf.reshape(img_ground_truth, [img_size * img_size, 3])
        )
    
    img_mask, img_train = get_img('TestCardF.jpg', 256)
    siren = Siren(in_features=2, out_features=3, hidden_features=256, hidden_layers=3, outermost_linear=True)


    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    optimizer = Adam()

    fig, axs = plt.subplots(4,4)
    j = 0
    k = 0

    for i in bar:
        with tf.GradientTape() as tape:
            img_mask = tf.cast(img_mask, dtype=tf.float32)
            img_train = tf.cast(img_train, dtype=tf.float32)
            y_batch = tf.constant(img_train)

            y_hat = siren(img_mask)
            loss = tf.math.reduce_mean(0.5* (y_batch - y_hat) ** 2)

        grads = tape.gradient(loss, siren.trainable_variables)
        optimizer.apply_gradients(grads, siren.trainable_variables)

        step_size *= decay_rate 

        if i % 10 == 0:
            if k == 4:
                j+=1
                k=0
            axs[j][k].imshow(((tf.reshape(y_hat, [256,256,3])*255).numpy().astype(np.uint8)))
            axs[j][k].axis('off')
            k+=1
            
        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()
        

    breakpoint()

    fig.savefig('preds.pdf')

    fig1, (axs1,axs2) = plt.subplots(1,2)
    axs1.axis('off')
    axs2.axis('off')
    fig1.suptitle(f"Loss:{loss.numpy():0.4f}, Iters:{num_iters}")
    axs1.imshow(((tf.reshape(y_batch, [256,256,3])*255).numpy().astype(np.uint8)))
    axs1.set_title("Original Image")
    axs2.imshow(((tf.reshape(y_hat, [256,256,3])*255).numpy().astype(np.uint8)))
    axs2.set_title(f"Predicted Image")
    fig1.savefig("finalpred.pdf")
    