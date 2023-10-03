import pytest
import tensorflow as tf

from cifar10 import Conv2d, Conv1x1, GroupNorm, ResidualBlock, Classifier


@pytest.mark.parametrize(
    'bs, height, width, channels',
    (
        (60, 32, 32, 3),
    ),
)
def test_dimensions_conv2d(bs, height, width, channels):
    conv2d = Conv2d(3, channels, width, 10, tf.nn.relu) # layer_kernel_sizes, input_depth, layer_depth, classes, activation
    tensor = tf.constant(
        tf.ones(shape = [bs, height, width, channels]),
        dtype=tf.dtypes.float32,
        name='tensor',
    )

    assert conv2d(tensor).shape == tf.TensorShape((bs, height, width, channels))


# @pytest.mark.parametrize(
#     'hidden_activation, output_activation',
#     (
#         (lambda x: tf.constant(0.0, shape=(1, 1)), lambda x: x),
#         (lambda x: x, lambda x: tf.constant(0.0, shape=(1, 1))),
#     ),
# )
# def test_activation(hidden_activation, output_activation):
#     mlp = MLP(
#         inputs=1,
#         outputs=1,
#         hidden_layers=1,
#         hidden_layer_width=1,
#         hidden_activation=hidden_activation,
#         output_activation=output_activation,
#     )

#     ones = tf.ones((1, 1))
#     zeros = tf.zeros((1, 1))

#     assert mlp(ones) == zeros