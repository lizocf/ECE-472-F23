# /bin/env python3.8

import pytest


def test_additivity():
    import tensorflow as tf

    from pset1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    basexp = BasisExpansion(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(basexp(a + b), basexp(a) + basexp(b), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from pset1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100

    basexp = BasisExpansion(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(basexp(a * b), basexp(a) * b, summarize=2)


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from pset1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    basexp = BasisExpansion(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = basexp(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    from pset1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    basexp = BasisExpansion(num_inputs, num_outputs, bias=bias)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = basexp(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, basexp.trainable_variables)

    for grad, var in zip(grads, basexp.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(basexp.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [([1000, 1000], [100, 100]), ([1000, 100], [100, 100]), ([100, 1000], [100, 100])],
)
def test_init_properties(a_shape, b_shape):
    import tensorflow as tf

    from pset1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape

    basexp_a = BasisExpansion(num_inputs_a, num_outputs_a, bias=False)
    basexp_b = BasisExpansion(num_inputs_b, num_outputs_b, bias=False)

    std_a = tf.math.reduce_std(basexp_a.w)
    std_b = tf.math.reduce_std(basexp_b.w)

    tf.debugging.assert_less(std_a, std_b)


def test_bias():
    import tensorflow as tf

    from pset1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    basexp_with_bias = BasisExpansion(1, 1, bias=True)
    assert hasattr(basexp_with_bias, "b")

    basexp_with_bias = BasisExpansion(1, 1, bias=False)
    assert not hasattr(basexp_with_bias, "b")