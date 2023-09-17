# /bin/env python3.8

import pytest


def test_additivity():
    import tensorflow as tf

    from ocfemia1 import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 10

    basexp = BasisExpansion(M)
    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(linear(basexp((a + b))), linear(basexp(a)) + linear((basexp(b))), summarize=2)

# Superposition should NOT hold

def test_homogeneity():
    import tensorflow as tf

    from ocfemia1 import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100
    M = 10

    basexp = BasisExpansion(M)
    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(linear(basexp(a * b)), linear(basexp(a)) * b, summarize=2)

# Homogeneity should NOT hold

@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from ocfemia1 import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    M = 10

    basexp = BasisExpansion(M)
    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, num_inputs])
    z = linear(basexp(a))

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    from ocfemia1 import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 10

    basexp = BasisExpansion(M)
    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = linear(basexp(a))
        loss = tf.math.reduce_mean(0.5*z**2)

    grads = tape.gradient(loss, linear.trainable_variables + basexp.trainable_variables)

    for grad, var in zip(grads, linear.trainable_variables + basexp.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(linear.trainable_variables + basexp.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1

# not too sure about this one :\


def test_bias():
    import tensorflow as tf

    from ocfemia1 import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    a = rng.normal(shape=[1, 1])

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    M = 10

    fit_with_bias = Linear(1, 1, M, bias=True)
    basexp = BasisExpansion(M)

    assert hasattr(fit_with_bias(basexp(a)), "b")

    fit_with_bias = Linear(1, 1, M, bias=False)
    assert not hasattr(fit_with_bias(basexp(a)), "b")

# or this :\\