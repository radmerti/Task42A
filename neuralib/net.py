import tensorflow as tf


class SinglePolyModel():
    """A multi-variate (input and ouput) polynomial regression model.

    Parameters
    ----------
    X : np.ndarray or tf.Variable
        Dataset for deriving the number of features.
    degrees : tuple, optional
        A list or tuple of polynomial degrees for the SinglePolyModel
        created, by default (2, 3, 4)
    out_dim : int, optional
        The output dimension of the model, by default 3
    """

    def __init__(self, X, degree: int = 1, out_dim: int = 3):
        self.degree = degree
        self.out_dim = out_dim
        self.trainable_variables = []

        init = tf.random_normal_initializer()
        self.trainable_variables = [
            tf.Variable(init(shape=(1, self.out_dim)), name="b", trainable=True)
        ]+[
            tf.Variable(init(shape=(X.shape[-1], self.out_dim)), name=f"W_{term_i}", trainable=True)
            for term_i in range(1, self.degree+1)
        ]

    def __call__(self, X):
        """
        Calls the forward pass of the model.

        Parameters
        ----------
        X : tf.Variable or np.ndarray
            The feature matrix.

        Returns
        -------
        tf.Tensor
            The output of the model.
        """
        term = self.trainable_variables[0]

        for term_i in range(1, self.degree+1):
            term = tf.add(
                tf.tensordot(
                    tf.pow(X, term_i),
                    self.trainable_variables[term_i],
                    axes=1),
                term)

        return term

class MultiPolyModel():
    """Model combining multiple SinglePolyModel to output multiple
    predictions.

    Parameters
    ----------
    X : np.ndarray or tf.Variable
        Dataset for deriving the number of features.
    degrees : tuple, optional
        A list or tuple of polynomial degrees for the SinglePolyModel
        created, by default (2, 3, 4)
    out_dim : int, optional
        The output dimension of the model, by default 3
    """

    def __init__(self, X, degrees: tuple = (2, 3, 4), out_dim: int = 3):
        self.degrees = degrees
        self.out_dim = out_dim

        self._singles = tuple(
            SinglePolyModel(X, degree=d, out_dim=out_dim)
            for d in degrees)

    @property
    def trainable_variables(self) -> [tf.Variable]:
        """
        Returns a list of trainable variables.

        Returns
        -------
        list of tf.Variable
            A list of tf.Variable that can be passed to the optimizer.
        """
        tvs = []
        for s in self._singles:
            tvs += s.trainable_variables
        return tvs

    def __call__(self, X) -> tf.Tensor:
        """
        Calls the forward pass of the model.

        Parameters
        ----------
        X : tf.Variable or np.ndarray
            The feature matrix.

        Returns
        -------
        tf.Tensor
            The output of the model.
        """
        return tf.stack(tuple(s(X) for s in self._singles), axis=1)
