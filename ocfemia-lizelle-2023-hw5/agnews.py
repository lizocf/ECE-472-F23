from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tensorflow as tf
    
# LOAD DATASET #
dataset = load_dataset("ag_news")

# trainText = dataset['train']['text'][0:10000]
# trainLabels = dataset['train']['label'][0:10000]

# valText = dataset['train']['text'][10000:12000]
# valLabels = dataset['train']['label'][10000:12000]

testText = dataset['test']['text']
testLabels = dataset['test']['label']

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# trainEmbeddings = model.encode(trainText)
# valEmbeddings = model.encode(valText)
testEmbeddings = model.encode(testText)

# breakpoint()

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
        
        self.num_hidden_layers = num_hidden_layers
    

        self.inputLayer = Linear(
            num_inputs = 384,
            num_outputs = hidden_layer_width
        )
        
        self.hiddenLayer = [
            Linear(num_inputs = hidden_layer_width,
                    num_outputs = hidden_layer_width)
            for i in range(num_hidden_layers)
            ]

        self.outputLayer = Linear(
            num_inputs = hidden_layer_width,
            num_outputs = 4
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
        # print(x.shape)

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

    # num_samples = len(trainEmbeddings)
    # num_samples = len(valEmbeddings)
    num_samples = len(testEmbeddings)
    num_inputs = 1
    num_outputs = 1

    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    refresh_rate = config["display"]["refresh_rate"]
    
    optimizer = Adam()

    def oneHotEncode(label):
        onehot = list()
        for value in label:
            row = np.zeros((4,))
            row[value] = 1.0
            label = onehot.append(row)
            # print(label)
        label = tf.cast(onehot, dtype = tf.float32)
        # breakpoint()
        return label

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, tf.nn.relu, tf.nn.sigmoid)

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            # x_batch = tf.gather(tf.Variable(trainEmbeddings), batch_indices)
            # y_batch = tf.reshape(tf.gather(oneHotEncode(trainLabels), batch_indices), (batch_size,4)) 

            # x_batch = tf.gather(tf.Variable(valEmbeddings), batch_indices)
            # y_batch = tf.reshape(tf.gather(oneHotEncode(valLabels), batch_indices), (batch_size,4)) 

            x_batch = tf.gather(tf.Variable(testEmbeddings), batch_indices)
            y_batch = tf.reshape(tf.gather(oneHotEncode(testLabels), batch_indices), (batch_size,4)) 

            y_hat = mlp(x_batch)

            # breakpoint()
            loss = tf.math.reduce_mean(-y_batch*tf.math.log(y_hat+(1e-7))-(1-y_batch)*tf.math.log(1-y_hat+(1e-7)))

        grads = tape.gradient(loss, mlp.trainable_variables)

        optimizer.apply_gradients(grads, mlp.trainable_variables)

        prediction = tf.math.argmax(y_hat, axis=-1)
        y_batch = tf.math.argmax(y_batch, axis=-1)
        equality = tf.math.equal(prediction, y_batch)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, Accuracy => {accuracy:.0%}, step_size => {step_size:0.4f}"
            )
            # if accuracy >= .955:
            #     with open('acc_loss_config.txt', 'a') as f:
            #         f.write(f"-----STEP_SIZE: {step_size}, BATCH_SIZE: {batch_size}, LAYER_DEPTH: {layer_depths} -----\n")
            #         f.write(f"Accuracy: {accuracy:.0%}. Loss: {loss}. Steps Taken: {i}. \n")
            #     exit()
            bar.refresh()