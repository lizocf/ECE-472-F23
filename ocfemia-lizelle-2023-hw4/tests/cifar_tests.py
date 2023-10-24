import pytest
import tensorflow as tf

from cifar10 import Conv2d, Conv1x1, GroupNorm, ResidualBlock, Classifier


@pytest.mark.parametrize(
    'bs, height, width, channels',
    (
        (60, 32, 32, 3),
        (100, 32, 32, 3),
    ),
)
def test_dims_inits_conv2d(bs, height, width, channels):
    layer_kernel_sizes = 3
    input_depth = channels
    num_classes = 10
    layer_depths = 50
    hidden_activation = tf.nn.relu
    conv2d = Conv2d(layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation) # layer_kernel_sizes, input_depth, layer_depth, classes, activation
    tensor = tf.constant(
        tf.ones(shape = [bs, height, width, channels]),
        dtype=tf.dtypes.float32,
        name='tensor',
    )

    assert conv2d.infilter.shape == tf.TensorShape((layer_kernel_sizes, layer_kernel_sizes, channels, layer_depths))
    assert conv2d.hidfilter.shape == tf.TensorShape((layer_kernel_sizes, layer_kernel_sizes, layer_depths, layer_depths))
    assert conv2d.ffilter.shape == tf.TensorShape((1, 1, layer_depths, input_depth))
    assert conv2d(tensor).shape == tf.TensorShape((bs, height, width, channels))

@pytest.mark.parametrize(
    'bs, height, width, channels',
    (
        (60, 32, 32, 3),
        (100, 32, 32, 3),
    ),
)
def test_dims_inits_conv1x1(bs, height, width, channels):
    layer_kernel_sizes = 3
    input_depth = channels
    num_classes = 10
    layer_depths = 50
    hidden_activation = tf.nn.relu
    conv1x1 = Conv1x1(layer_kernel_sizes, input_depth, layer_depths, num_classes, hidden_activation) # layer_kernel_sizes, input_depth, layer_depth, classes, activation
    tensor = tf.constant(
        tf.ones(shape = [bs, height, width, channels]),
        dtype=tf.dtypes.float32,
        name='tensor',
    )

    assert conv1x1.ffilter.shape == tf.TensorShape((1,1,input_depth, num_classes))
    assert conv1x1(tensor).shape == tf.TensorShape((bs, 1, 1, num_classes))

@pytest.mark.parametrize(
    'bs, height, width, channels',
    (
        (60, 32, 32, 3),
        (100, 32, 32, 3),
    ),
)
def test_dims_inits_groupnorm(bs, height, width, channels):

    gnorm = GroupNorm()# layer_kernel_sizes, input_depth, layer_depth, classes, activation
    tensor = tf.constant(
        tf.ones(shape = [bs, height, width, channels]),
        dtype=tf.dtypes.float32,
        name='tensor',
    )

    ones = tf.ones((1,channels,1,1))
    assert gnorm.gamma.shape == ones.shape
    assert gnorm.beta.shape == ones.shape
    
    assert gnorm(tensor).shape == tf.TensorShape((bs, height, width, channels))


@pytest.mark.parametrize(
    'bs, height, width, channels',
    (
        (60, 32, 32, 3),
        (100, 32, 32, 3),
    ),
)
def test_dims_inits_resblock(bs, height, width, channels):
    layer_kernel_sizes = 3
    input_depth = channels
    num_classes = 10
    layer_depths = 50
    hidden_activation = tf.nn.relu
    
    resblock = ResidualBlock(layer_kernel_sizes,input_depth,layer_depths,num_classes, hidden_activation)# layer_kernel_sizes, input_depth, layer_depth, classes, activation
    
    tensor = tf.constant(
        tf.ones(shape = [bs, height, width, channels]),
        dtype=tf.dtypes.float32,
        name='tensor',
    )
    
    assert resblock(tensor).shape == tf.TensorShape((bs, height, width, channels))


@pytest.mark.parametrize(
    'bs, height, width, channels',
    (
        (60, 32, 32, 3),
        (100, 32, 32, 3),
    ),
)
def test_dims_inits_resblock(bs, height, width, channels):
    layer_kernel_sizes = 3
    input_layer = channels
    num_classes = 10
    layer_depths = 50
    hidden_activation = tf.nn.relu
    num_res_blocks = [0,2,2,2,2]
    
    classifier = Classifier(input_layer, layer_depths, layer_kernel_sizes, num_classes, num_res_blocks, hidden_activation)# layer_kernel_sizes, input_depth, layer_depth, classes, activation
    
    tensor = tf.constant(
        tf.ones(shape = [bs, height, width, channels]),
        dtype=tf.dtypes.float32,
        name='tensor',
    )
    
    assert classifier(tensor).shape == tf.TensorShape((bs, num_classes))
