import tensorflow as tf
import numpy as np

def integrated_gradients(model, X_test, baseline_recurrent=None, steps=50):
    """
    Compute Integrated Gradients for the given model and inputs.

    Args:
        model: Trained TensorFlow model.
        X_test: Test recurrent input (shape: batch_size, time_steps, features).
        baseline_recurrent: Baseline input for recurrent features (default: all zeros).
        baseline_static: Baseline input for static features (default: all zeros).
        steps: Number of steps to approximate the integral.

    Returns:
        IG for recurrent and static features.
    """
    if baseline_recurrent is None:
        baseline_recurrent = np.zeros_like(X_test)


    alphas = np.linspace(0, 1, steps + 1)  # Shape: (steps+1,)

    gradients_recurrent = []

    for alpha in alphas:
        interpolated_recurrent = baseline_recurrent + alpha * (X_test - baseline_recurrent)

        # Convert to TensorFlow Variables
        interpolated_recurrent_tf = tf.Variable(interpolated_recurrent, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:  # Set persistent=True to allow multiple gradient calculations
            tape.watch(interpolated_recurrent_tf)

            # Model predictions for this interpolated input
            predictions = model((interpolated_recurrent_tf), training=False)
            annual_revenue = tf.reduce_sum(predictions, axis=1)  # Sum over months

        # Compute gradients for this step
        grad_recurrent = tape.gradient(annual_revenue, interpolated_recurrent_tf)

        gradients_recurrent.append(grad_recurrent.numpy())

    # Convert gradients list to numpy arrays
    gradients_recurrent = np.array(gradients_recurrent)  # Shape: (steps+1, batch, time, features)

    # Average gradients across steps to approximate integral
    avg_grad_recurrent = np.mean(gradients_recurrent, axis=0)

    # Compute IG: (input - baseline) * average gradient
    IG_recurrent = (X_test - baseline_recurrent) * avg_grad_recurrent

    return IG_recurrent