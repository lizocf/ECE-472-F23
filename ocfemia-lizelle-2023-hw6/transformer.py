from datasets import load_dataset
import tensorflow as tf
import einops
import numpy as np
    
def tokenize(sequence):
# remove punctuation
    for punc in ["!", ".", ",","?"]:
        # breakpoint()
        sequence = sequence.replace(punc, "") # replace punctuation
    return [token.lower() for token in sequence.split(" ")] # lowercase

def build_vocab(data):
    vocab = list(set(tokenize(data)))
    vocab.sort()
    stoi = {word:i for i, word in enumerate(vocab)}  # assign an integer to each word
    return stoi

class Embedding(tf.Module):
    def __init__(self, sequence, d_model, stoi):
        rng = tf.random.get_global_generator()
        self.tokenized_sequences = [tokenize(seq) for seq in sequence]
        self.indexed_sequences = tf.constant([[stoi[word] for word in seq] for seq in self.tokenized_sequences])
        vocab_size = len(stoi)
        
        self.embeddings = tf.Variable(
            rng.normal(shape=[vocab_size, d_model]), 
            trainable=True,
            name="embedding") # did not need to be trained? When added, grad = None.

        self.embedded_seq = tf.gather(self.embeddings,self.indexed_sequences)
        
    def __call__(self):
        return self.embedded_seq


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=False):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs)) 

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


class FFN(tf.Module):
    def __init__(self, d_model, d_ffn, num_hidden_layers):
        self.w1 = Linear(d_model, d_ffn)
        self.w2 = Linear(d_ffn, d_model)

    def __call__(self, x):
        z = self.w1(x)
        z = tf.nn.relu(z)
        z = self.w2(tf.nn.dropout(z,rate=0.5))
        return z

class MultiHeadAttention(tf.Module):
    def __init__(self, num_heads, d_model):
        self.d_k = d_model//num_heads
        self.d_v = d_model//num_heads

        self.n_heads = num_heads

        self.wq = Linear(d_model,d_model)
        self.wk = Linear(d_model,d_model)
        self.wv = Linear(d_model,d_model)
        self.wo = Linear(d_model,d_model)

    def __call__(self, sequence, mask: tf.Tensor = None):

        q = self.wq(sequence)
        k = self.wk(sequence)
        v = self.wv(sequence)

        batch_size = q.shape[0]

        # split into n-heads
        Q = tf.transpose(tf.reshape(q, (batch_size, -1, self.n_heads, self.d_k)), (0,2,1,3)) # [bs,len,dm] -> [bs,len,nh,dk]
        K = tf.transpose(tf.reshape(k, (batch_size, -1, self.n_heads, self.d_k)), (0,2,1,3)) # [bs,len,dm] -> [bs,len,nh,dk]        
        V = tf.transpose(tf.reshape(v, (batch_size, -1, self.n_heads, self.d_k)), (0,2,1,3)) # [bs,len,dm] -> [bs,len,nh,dk]
        
        # SDP = QK^T
        scaled_dot_prod = einops.einsum(Q, K, 'b s i k, b s j k -> b s i k')/np.sqrt(self.d_k)

        if mask is not None:
            scaled_dot_prod += mask

        attention = tf.nn.softmax(scaled_dot_prod,-1)

        A = einops.einsum(attention, V, 'b s i k, b s j k  -> b s i k')
        A = tf.reshape((tf.transpose(A, (0,2,1,3))), (batch_size, -1, self.n_heads*self.d_k))

        output = self.wo(A)

        return output, attention


class TransformerDecoder(tf.Module):
    def __init__(self, bs, seq_length, d_model, n_layers, n_heads, d_ffn, vocab_size):
        self.attention = MultiHeadAttention(n_heads, d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=[1,2]) # I tried using GroupNorm from last hw, but shape issues :\
        self.ffn = FFN(d_model, d_ffn, n_layers)
        self.linear = Linear(d_model, vocab_size)

    def __call__(self, sequence, mask):
        z, z_prob = self.attention(sequence, mask)
        z = self.layernorm(z)
        z = self.ffn(z)

        z = self.linear(z)
        z = tf.nn.softmax(z)
        return z, z_prob

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
        prog="TransformerDecoder",
        description="decodes sequence given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    optimizer = Adam()

##### NEW TF STUFF ######

    d_model = config["tf"]["d_model"]
    num_heads = config["tf"]["num_heads"]
    num_layers = config["tf"]["num_layers"]

    # My TF decoder is loosely based off of https://medium.com/@hunter-j-phillips/the-decoder-8882c33de69a.
    # This ``example`` is used to create a simply dictionary
    example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses? <beg> <end> <pad>"

    sequence = ["<beg> I wonder what will come next! <end>"] # INPUT
    trg_sequence = ["I wonder what will come next! <end> <pad>"] # LABEL (only tested on one shift)

    stoi = build_vocab(example)

    embed = Embedding(sequence, d_model, stoi)
    tr_embed = Embedding(trg_sequence, d_model, stoi)

    embedded_seq = embed()
    embedded_trg = tr_embed()

    vocab_size = len(stoi)
    itos = {v:k for k,v in stoi.items()}

    batch_size = embedded_seq.shape[0]
    seq_length = embedded_seq.shape[1]    

    mask = tf.fill([batch_size,num_heads, seq_length, d_model//num_heads], float('-inf'))
    mask = tf.linalg.band_part(mask,0,-1)
    mask = tf.linalg.set_diag(mask,tf.zeros([batch_size, num_heads, seq_length]))

    decode = TransformerDecoder(batch_size, seq_length, d_model, num_layers, num_heads, d_model*4, vocab_size)

    bar = trange(num_iters) 
    predictions = []
    for i in bar:
        with tf.GradientTape(persistent=True) as tape:
            embedded_seq = embed() # embed.trainvals have no effect on loss
            y_hat, y_prob = decode(embedded_seq, mask)
            y_hat = tf.cast(y_hat, dtype='float32')
            y_batch = tf.cast(tr_embed.indexed_sequences, dtype='int32')

            loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.squeeze(y_batch), logits = tf.squeeze(y_hat)))
            # had an issue using non-sparse softmax where gradients = None for every trainable variable. Kind of resolved by using
            # sparse softmax, but the only issue is that I can only train on one batch_size now, i.e, multiple sentence input can't
            # be trained... 

        grads = tape.gradient(loss, decode.trainable_variables)

        optimizer.apply_gradients(grads, decode.trainable_variables)

        step_size *= decay_rate 
        predictions.append([itos[idx] for idx in tf.math.argmax(y_hat, axis=-1).numpy()[0]])

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    print(*predictions, sep='\n')  # print out predictions per iter :D

    