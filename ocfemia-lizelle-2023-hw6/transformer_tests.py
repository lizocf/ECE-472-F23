import pytest
import tensorflow as tf
import numpy as np

from transformer import tokenize, build_vocab, Embedding, MultiHeadAttention, Linear, FFN, TransformerDecoder

@pytest.mark.parametrize(
    'sentence',
    (
        "Uh oh, what is happening?",
        "Cheers! This is an example!",
        "Gosh, this is not an example."
    ),
)
def test_build_vocab(sentence):
    vocab = build_vocab(sentence)
    assert len(vocab) == len(sentence.split())

@pytest.mark.parametrize(
    'sentence',
    (
        ["I wonder what will come next!",
         "This is a basic example paragraph.",
         "Hello, what is a basic split?"],
    )
)
def test_embedding(sentence):
    example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"
    d_model = 3
    stoi = build_vocab(example)
    embed = Embedding(sentence, d_model, stoi)
    
    assert embed().shape == [len(sentence), len(sentence[0].split()),d_model]
    # assert embed()

# ^ works in transformer.py, not here tho ):

@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs, bias=bias)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = linear(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, linear.trainable_variables)

    # breakpoint()
    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(linear.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1


@pytest.mark.parametrize(
    'bs, seq_length, d_model, num_heads',
    (
        (1,4,512,8),
    ),
)
def test_multihead(bs, seq_length, num_heads, d_model):
    # print(num_heads, d_model)
    multihead = MultiHeadAttention(num_heads, d_model)
    # breakpoint()
    # tensor = tf.random.uniform((bs, seq_length, input_dim),dtype = tf.int32)
    x = tf.random.uniform((bs, seq_length, d_model), dtype=tf.float32)
    y = tf.random.uniform((bs, seq_length, d_model), dtype=tf.float32)
    
    mask = tf.fill([bs,num_heads, seq_length, d_model//num_heads], float('-inf'))
    mask = tf.linalg.band_part(mask,0,-1)
    mask = tf.linalg.set_diag(mask,tf.zeros([bs, num_heads, seq_length]))

    with tf.GradientTape() as tape:
        tape.watch(multihead.trainable_variables)
        out = multihead(x)
        loss = tf.math.reduce_mean(out[0]**2)
        loss = tf.constant(11.11)
        grads = tape.gradient(loss, multihead.trainable_variables)

    assert len(grads) == len(multihead.trainable_variables)
    assert out[0].shape == [bs, seq_length, d_model]
    # assert 

@pytest.mark.parametrize(
    'bs, seq_length, d_model, num_heads, vocab_size',
    (
        (1,4,512,8,27),
    ),
)
def test_transformer(bs, seq_length, num_heads, d_model, vocab_size):
    # print(num_heads, d_model)
    decode = TransformerDecoder(bs, seq_length, d_model, 8, 8, d_model*4, vocab_size)
    x = tf.random.uniform((1,seq_length, d_model), dtype='float32')
    y = tf.random.uniform(shape = [seq_length], maxval=vocab_size, dtype='int32')
    # breakpoint()
    mask = tf.fill([bs,num_heads, seq_length, d_model//num_heads], float('-inf'))
    mask = tf.linalg.band_part(mask,0,-1)
    mask = tf.linalg.set_diag(mask,tf.zeros([bs, num_heads, seq_length]))

    with tf.GradientTape() as tape:
        y_hat, y_prob = decode(x, mask)  
        y_hat = tf.cast(y_hat, dtype='float32')

        loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = tf.squeeze(y_hat)))
        grads = tape.gradient(loss, decode.trainable_variables)
    
    assert y_hat.shape == [bs, seq_length, vocab_size]
    assert len(grads) == len(decode.trainable_variables)
    assert np.allclose(y_prob, np.tril(y_prob)) # check if attention matrix is lower triangular, i.e. does not look to future words for predictions :)


