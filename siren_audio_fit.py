#!/bin/env python
# super radical interesting thing you can do with siren :pog:

import tensorflow as tf
import numpy as np
from siren_img_fit import Linear, SineLayer, Siren, Adam 


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
    import tensorflow_io as tfio

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


    def get_aud(aud_path):
        aud_raw = tf.io.read_file(aud_path)
        waveform = tfio.audio.decode_wav(aud_raw, dtype=tf.float32)
        return (
            get_mgrid(len(waveform), 1),
            waveform
        )

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

    aud_mask, aud_train = get_aud('gt_bach.wav')
    
    audio_siren = Siren(in_features=1, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
    

    for i in bar:
        with tf.GradientTape() as tape:
            aud_mask = tf.cast(aud_mask, dtype=tf.float32)
            aud_train = tf.cast(aud_train, dtype=tf.float32)
            y_batch = tf.constant(aud_train)

            y_hat = audio_siren(aud_mask)
            loss = tf.math.reduce_mean(0.5* (y_batch - y_hat) ** 2)

        grads = tape.gradient(loss, siren.trainable_variables)
        optimizer.apply_gradients(grads, siren.trainable_variables)

        step_size *= decay_rate 

        if i % 10 == 0:
            if k == 4:
                j+=1
                k=0
            axs[j][k].imshow(y_hat.numpy().squeeze())
            k+=1

        breakpoint()

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()
        
    # fig.savefig('preds.pdf')

    # fig1, (axs1,axs2) = plt.subplots(1,2)
    # axs1.axis('off')
    # axs2.axis('off')
    # fig1.suptitle(f"Loss:{loss.numpy():0.4f}, Iters:{num_iters}")
    # axs1.imshow(((tf.reshape(y_batch, [256,256,3])*255).numpy().astype(np.uint8)))
    # axs1.set_title("Original Image")
    # axs2.imshow(((tf.reshape(y_hat, [256,256,3])*255).numpy().astype(np.uint8)))
    # axs2.set_title(f"Predicted Image")
    # fig1.savefig("finalpred.pdf")
    