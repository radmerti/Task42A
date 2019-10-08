import tensorflow as tf

def mean_squared_error(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """Computes the means squared error between two rank-L tensors,
    where the first dimension is the batch-dimension.

    Parameters
    ----------
    prediction : tf.Tensor
        A rank-3 tensor with dimensions (M, K, L). The K predictions
        for M samples of L-dimensional vectors in a batch.
    target : tf.Tensor
        A rank-2 tensor with dimensions (M, L). The target values
        for M predictions of L-dimensional vectors in a batch.

    Returns
    -------
    tf.Tensor
        A rank-1 tensor with dimensions (M) of losses for each sample
        in the batch.
    """
    assert len(prediction.shape) == 3, "'prediction' rank must be 3"
    assert len(target.shape) == 2, "'target' rank must be 2"
    assert prediction.shape[0] == target.shape[0], (
        f"batch sizes must match, "
        f"{prediction.shape[0]} != {target.shape[0]}")
    assert prediction.shape[-1] == target.shape[-1], (
        f"target vector sizes must match, "
        f"{prediction.shape[-1]} != {target.shape[-1]}")

    ident = tf.ones((prediction.shape[1], prediction.shape[-1]))
    casted = tf.einsum('ml,kl->mkl', target, ident)
    sq = tf.square(prediction-casted)
    m1 = tf.reduce_mean(sq, axis=2)
    m2 = tf.reduce_mean(m1, axis=1)
    return m2


def _main():
    prediction = tf.Variable(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    target = tf.Variable(
        [[-2.0, -2.0, -2.0],
         [-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [3.0, 3.0, 3.0]])

    loss = mean_squared_error(prediction, target)

    print(f"loss={loss}")

if __name__ == "__main__":
    _main()
