import tensorflow as tf
import numpy as np

def integrated_gradients(model, X_test, X_static_test, baseline=None, steps=50):
    """
    Compute Integrated Gradients for LSTM model.

    Args:
        model: Trained TensorFlow model.
        X_test: Recurrent input (batch_size, time_steps, features).
        X_static_test: Static input (batch_size, static_features).
        baseline: Reference baseline input (default is zero array).
        steps: Number of steps for interpolation.

    Returns:
        IG_recurrent: Integrated gradients for recurrent features.
        IG_static: Integrated gradients for static features.
    """

    if baseline is None:
        baseline = np.zeros_like(X_test)

    interpolated_recurrent = np.linspace(baseline, X_test, num=steps)
    interpolated_static = np.linspace(np.zeros_like(X_static_test), X_static_test, num=steps)

    interpolated_recurrent_tf = tf.convert_to_tensor(interpolated_recurrent, dtype=tf.float32)
    interpolated_static_tf = tf.convert_to_tensor(interpolated_static, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(interpolated_recurrent_tf)
        tape.watch(interpolated_static_tf)

        predictions = model((interpolated_recurrent_tf, interpolated_static_tf), training=False)
        target_output = tf.reduce_sum(predictions, axis=1)

    gradients_recurrent = tape.gradient(target_output, interpolated_recurrent_tf)
    gradients_static = tape.gradient(target_output, interpolated_static_tf)

    IG_recurrent = np.mean(gradients_recurrent.numpy(), axis=0) * (X_test - baseline)
    IG_static = np.mean(gradients_static.numpy(), axis=0) * X_static_test

    return IG_recurrent, IG_static