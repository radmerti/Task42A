# Multiple Prediction Regression Loss

A test of a multi-regression mean-squared-loss that takes a prediction
tensor of the form (M, K, L), where M is the batch-size, K is the number
of prediction per sample of the model, and L is the single sample
output-vector-size. The target value is a tensor of the form (M, L). The loss is defined in the function `mean_squared_error()` in [neuralib/loss.py](neuralib/loss.py).

To test the loss a simple multi-regression model is provided. It fits multiple polynomial regressions to the data. It returns the concatenated output of the individual models as a tensor of the form (M, K, L). The dimensions are defined as above for the loss. The model is defined in the class `MultiPolyModel()` in [neuralib/net.py](neuralib/net.py).


To run, first run

```bash
# recommended to run everything in a virtualenv
pip install -r requirements.txt
```

Then run

```bash
python train.py
```

This will train a simple neural network that fits multiple polynomial
models to the data using the multiple output regression loss.

To only run a test of the loss, use

```bash
python neuralib/loss.py
```