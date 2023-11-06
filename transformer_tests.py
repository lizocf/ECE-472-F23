import pytest
import tensorflow as tf

from transformer import tokenize, build_vocab, positional_encoding, Embedding, MultiHeadAttention, Linear, FFN


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

# @pytest.mark.parametrize(
#     'sentence',
#     (
#         [["I wonder what will come next!"],
#          ["This is a basic example paragraph."],
#          ["Hello, what is a basic split?"]]
#     ),
# )
# def test_embedding(sentence):
#     example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"
#     d_model = 3
#     stoi = build_vocab(example)

#     # breakpoint()
#     embed = Embedding(sentence, d_model, stoi)
#     breakpoint()
#     assert embed().shape[0] == [len(sentence), len(sentence.split()),d_model]

# ^ works in transformer.py, not here tho ):

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
    
    mask = tf.fill([bs,num_heads, seq_length, d_model//num_heads], float('-inf'))
    mask = tf.linalg.band_part(mask,0,-1)
    mask = tf.linalg.set_diag(mask,tf.zeros([bs, num_heads, seq_length]))

    out = multihead(x)
    out_mask = multihead(x, mask)
    assert out[0].shape == [bs, seq_length, d_model]
    # assert 

# @pytest.mark.parametrize(
#     'd_model, d_ffn',
#     (
#         (512, 4*512),
#     ),
# )
# def test_ffn(d_model, d_ffn):
#     # print(num_heads, d_model)
#     ffn = FFN(d_model, d_ffn)

#     out = ffn()

#     assert out.shape == [bs, seq_length, d_model]
#     # assert 



# @pytest.mark.parametrize(
#     'bs, height, width, channels',
#     (
#         (60, 32, 32, 3),
#         (100, 32, 32, 3),
#     ),
# )
# def test_dims_inits_conv1x1(bs, height, width, channels):
#     layer_kernel_sizes = 3
#     input_depth = channels
#     num_classes = 10
#     layer_depths = 50
#     hidden_activation = tf.nn.relu
#     conv1x1 = Conv1x1(layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation) # layer_kernel_sizes, input_depth, layer_depth, classes, activation
#     tensor = tf.constant(
#         tf.ones(shape = [bs, height, width, channels]),
#         dtype=tf.dtypes.float32,
#         name='tensor',
#     )

#     assert conv1x1.ffilter.shape == tf.TensorShape((1,1,input_depth, num_classes))
#     assert conv1x1(tensor).shape == tf.TensorShape((bs, 1, 1, num_classes))

# @pytest.mark.parametrize(
#     'bs, height, width, channels',
#     (
#         (60, 32, 32, 3),
#         (100, 32, 32, 3),
#     ),
# )
# def test_dims_inits_groupnorm(bs, height, width, channels):

#     gnorm = GroupNorm()# layer_kernel_sizes, input_depth, layer_depth, classes, activation
#     tensor = tf.constant(
#         tf.ones(shape = [bs, height, width, channels]),
#         dtype=tf.dtypes.float32,
#         name='tensor',
#     )

#     ones = tf.ones((1,channels,1,1))
#     assert gnorm.gamma.shape == ones.shape
#     assert gnorm.beta.shape == ones.shape
    
#     assert gnorm(tensor).shape == tf.TensorShape((bs, height, width, channels))


# @pytest.mark.parametrize(
#     'bs, height, width, channels',
#     (
#         (60, 32, 32, 3),
#         (100, 32, 32, 3),
#     ),
# )
# def test_dims_inits_resblock(bs, height, width, channels):
#     layer_kernel_sizes = 3
#     input_depth = channels
#     num_classes = 10
#     layer_depths = 50
#     hidden_activation = tf.nn.relu
    
#     resblock = ResidualBlock(layer_kernel_sizes,input_depth,layer_depths,num_classes, hidden_activation)# layer_kernel_sizes, input_depth, layer_depth, classes, activation
    
#     tensor = tf.constant(
#         tf.ones(shape = [bs, height, width, channels]),
#         dtype=tf.dtypes.float32,
#         name='tensor',
#     )
    
#     assert resblock(tensor).shape == tf.TensorShape((bs, height, width, channels))


# @pytest.mark.parametrize(
#     'bs, height, width, channels',
#     (
#         (60, 32, 32, 3),
#         (100, 32, 32, 3),
#     ),
# )
# def test_dims_inits_resblock(bs, height, width, channels):
#     layer_kernel_sizes = 3
#     input_layer = channels
#     num_classes = 10
#     layer_depths = 50
#     hidden_activation = tf.nn.relu
#     num_res_blocks = [0,2,2,2,2]
    
#     classifier = Classifier(input_layer, layer_depths, layer_kernel_sizes, num_classes, num_res_blocks, hidden_activation)# layer_kernel_sizes, input_depth, layer_depth, classes, activation
    
#     tensor = tf.constant(
#         tf.ones(shape = [bs, height, width, channels]),
#         dtype=tf.dtypes.float32,
#         name='tensor',
#     )
    
#     assert classifier(tensor).shape == tf.TensorShape((bs, num_classes))
