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

    tf.debugging.assert_near(basexp((a + b)), basexp(a) + (basexp(b)), summarize=2)

## Superposition does NOT hold

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

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(basexp(a * b), basexp(a) * b, summarize=2)

## Homogeneity does NOT hold either

## -> Base expansion is NOT linear :)

@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from ocfemia1 import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    M = 10

    basexp = BasisExpansion(M)

    a = rng.normal(shape=[1, num_inputs])
    z = basexp(a)

    tf.assert_equal(tf.shape(z)[-1], M)

## tf.shape(z)[-1] should be equal to M
