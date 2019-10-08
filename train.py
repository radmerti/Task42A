import numpy as np
import tensorflow as tf

from neuralib.loss import mean_squared_error
from neuralib.net import MultiPolyModel


def train(model: callable, dataset: callable, optimizer: tf.keras.optimizers.Optimizer, n_epochs: int = 5):
    """
    Trains a model.

    Parameters
    ----------
    model : callable
        Model must be callable as f(X) where X is the feature matrix
        for a batch.
    dataset : callable
        A method that returns an iterable when called. The iterable
        yields batches of X, y, e.g. the feature matrix and target.
    optimizer : tf.keras.optimizers.Optimizer
        An optimizer instance that supports the apply_gradients
        method.
    n_epochs : int, optional
        Number of epochs, by default 5
    """
    for epoch_i in range(n_epochs):
        print(f"epoch: {epoch_i+1:>4}", end='')
        epoch_loss = 0.0
        epoch_steps = 0
        for x, y in dataset():
            with tf.GradientTape() as tape:
                prediction = model(x)
                loss = mean_squared_error(prediction, y)
                epoch_loss += np.mean(loss)
                epoch_steps += 1
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f", loss={epoch_loss/epoch_steps:>9.2f}")


def _main():
    X = np.hstack([
        np.linspace(0, np.pi, 100, dtype='float32').reshape(-1, 1),
        np.linspace(np.pi/2, np.pi*(3/2), 100, dtype='float32')[::-1].reshape(-1, 1)])
    Y = np.hstack([
        np.cos(X[:, 1]).reshape(-1, 1),
        np.cos(X[:, 0]).reshape(-1, 1),
        -np.sin(X[:, 1]).reshape(-1, 1)])

    def dataset(batch_size: int = 10):
        # shuffle the dataset on each call (epoch)
        p = np.random.permutation(len(X))
        X[:, :] = X[p]
        Y[:, :] = Y[p]
        for x, y in zip(
                X.reshape(-1, batch_size, X.shape[-1]),
                Y.reshape(-1, batch_size, Y.shape[-1])):
            yield (tf.Variable(x), tf.Variable(y))


    train(MultiPolyModel(X),
          dataset,
          tf.keras.optimizers.Adam(learning_rate=0.001),
          n_epochs=100)

if __name__ == "__main__":
    _main()
