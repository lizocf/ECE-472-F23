from datasets import load_dataset
# from sentence_transformers import SentenceTransformer
import tensorflow as tf
import einops
import numpy as np
    
class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=False):
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


def positional_encoding(seq_length, d_model): # https://www.tensorflow.org/text/tutorials/transformer
    depth = depth/2

    positions = np.arange(seq_length)[:,np.newaxis]         # (seq,1)
    depths = np.arange(d_model)[np.newaxis, :]/d_model      # (1,d_model)

    angle_rates = 1 / (10000**d_model)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class Embedding(tf.Module):
    def __init__(self, sequence, d_model, stoi, label = False):
        rng = tf.random.get_global_generator()
        self.tokenized_sequences = [tokenize(seq) for seq in sequence]
        # breakpoint()
        self.indexed_sequences = tf.constant([[stoi[word] for word in seq] for seq in self.tokenized_sequences])
        vocab_size = len(stoi)
        
        embeddings = tf.Variable(rng.normal(shape=[vocab_size, d_model]))
        self.embedded_seq = tf.gather(embeddings,self.indexed_sequences)
        
        # trg = []
        # for i in self.tokenized_sequences:
        #     y = i
        #     trg.append(y[1:].append('<pad>'))
        
        # self.indexed_trg = tf.constant([[stoi[word] for word in seq] for seq in trg])
        
        # self.trg_seq = tf.gather(embeddings, self.indexed_trg) 

    def __call__(self):
        return self.embedded_seq
        # breakpoint()
   

tensor = tf.random.uniform((1, 4, 512),dtype = tf.float32)
query = tf.random.uniform((1, 4, 512),dtype = tf.float32)
key = tf.random.uniform((1, 4, 512),dtype = tf.float32)
value = tf.random.uniform((1, 4, 512),dtype = tf.float32)

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
        
# multi = MultiHeadAttention(8,512) && multi(query,key,value)
# breakpoint()

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

    def __call__(self, input: tf.Tensor, mask: tf.Tensor = None):


        Q = self.wq(input)
        K = self.wk(input)
        V = self.wv(input)

        batch_size = Q.shape[0]

        # split into n-heads
        Q = tf.transpose(tf.reshape(Q, (batch_size, -1, self.n_heads, self.d_k)), (0,2,1,3)) # [bs,len,dm] -> [bs,len,nh,dk]
        K = tf.transpose(tf.reshape(K, (batch_size, -1, self.n_heads, self.d_k)), (0,2,1,3)) # [bs,len,dm] -> [bs,len,nh,dk]        
        V = tf.transpose(tf.reshape(V, (batch_size, -1, self.n_heads, self.d_k)), (0,2,1,3)) # [bs,len,dm] -> [bs,len,nh,dk]
        # SDP = QK^T
        scaled_dot_prod = einops.einsum(Q, K, 'b s i k, b s j k -> b s i k')/np.sqrt(self.d_k)

        if mask is not None:
            scaled_dot_prod += mask

        attention = tf.nn.softmax(scaled_dot_prod,-1)
        # print(attention)

        A = einops.einsum(attention, V, 'b s i k, b s j k  -> b s i k')
        A = tf.reshape((tf.transpose(A, (0,2,1,3))), (batch_size, -1, self.n_heads*self.d_k))

        output = self.wo(A)
        # breakpoint()
        return output, attention



class TransformerDecoder(tf.Module):
    def __init__(self, bs, seq_length, d_model, n_layers, n_heads, d_ffn):
        self.attention = MultiHeadAttention(n_heads, d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=[1,2])
        self.ffn = FFN(d_model, d_ffn, n_layers)
        self.linear = Linear(d_model, d_model)

        # print(mask)
    
    def __call__(self, sequence, mask):
        for i in range(8):
            z, z_prob = self.attention(sequence, mask)
            z = self.layernorm(z)
            z = self.ffn(z)
        z = tf.nn.relu(z)
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


# breakpoint()

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

    example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses? <beg> <end> <pad>"

    sequence = ["<beg> I wonder what will come next! <end>"]

    trg_sequence = ["I wonder what will come next! <end> <pad>"]

    stoi = build_vocab(example)

    embed = Embedding(sequence, d_model, stoi)
    tr_embed = Embedding(trg_sequence, d_model, stoi)
    # breakpoint()
    embedded_seq = embed()
    embedded_trg = tr_embed()

    # breakpoint()


    vocab_size = len(stoi)
        
    batch_size = embedded_seq.shape[0]
    seq_length = embedded_seq.shape[1]    


    mask = tf.fill([batch_size,num_heads, seq_length, d_model//num_heads], float('-inf'))
    mask = tf.linalg.band_part(mask,0,-1)
    mask = tf.linalg.set_diag(mask,tf.zeros([batch_size, num_heads, seq_length]))


    # breakpoint()

    bar = trange(num_iters)
    # batch_indices = rng.uniform(
    #         shape=[batch_size], maxval=num_samples, dtype=tf.int32
    #     )

    for i in bar:
        with tf.GradientTape() as tape:
            # x_batch = tf.gather(tf.Variable(testEmbeddings), batch_indices)
            # y_batch = tf.reshape(tf.gather(oneHotEncode(testLabels), batch_indices), (batch_size,4)) 
            decode = TransformerDecoder(batch_size, seq_length, d_model, num_layers, num_heads, d_model*4)
            
            y_hat, y_prob = decode(embedded_seq, mask)  
            y_batch = embedded_trg

            print(y_hat[0])
            loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_batch, logits = y_hat))

        grads = tape.gradient(loss, decode.trainable_variables)

        optimizer.apply_gradients(grads, decode.trainable_variables)

        # prediction = tf.math.argmax(y_hat, axis=-1)
        # y_batch = tf.math.argmax(y_batch, axis=-1)
        # equality = tf.math.equal(prediction, y_batch)
        # accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

        step_size *= decay_rate 

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss}, step_size => {step_size:0.4f}"
            )
            # if accuracy >= .955:
            #     with open('acc_loss_config.txt', 'a') as f:
            #         f.write(f"-----STEP_SIZE: {step_size}, BATCH_SIZE: {batch_size}, LAYER_DEPTH: {layer_depths} -----\n")
            #         f.write(f"Accuracy: {accuracy:.0%}. Loss: {loss}. Steps Taken: {i}. \n")
            #     exit()
            bar.refresh()

            # docker run --gpus all -it --rm -v $(pwd):/workdir tensorflow/tensorflow:latest-gpu bash
