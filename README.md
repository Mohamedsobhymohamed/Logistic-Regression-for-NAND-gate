# Logistic Regression for NAND Gate

## Overview
This project implements a simple logistic regression model from scratch to learn the NAND gate function using Python and NumPy. The model is trained using gradient descent and evaluated based on its accuracy and loss.

## Features
- Implements logistic regression using a custom forward pass and gradient computation.
- Uses the sigmoid activation function for binary classification.
- Computes the binary cross-entropy loss.
- Updates weights and bias using gradient descent.
- Trains on the NAND gate dataset.
- Visualizes training loss and accuracy.
- Plots decision boundary.
- Includes test cases for validation.

## Dependencies
Make sure you have the following dependencies installed:
```bash
pip install numpy matplotlib
```

## Usage
Run the script to train the model and visualize results:
```bash
python logistic_regression_nand.py
```

## Explanation of Key Components

### Sigmoid Activation Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
The sigmoid function maps any input to a value between 0 and 1, making it useful for binary classification.

### Forward Pass
```python
def forward_pass(x1, x2, w1, w2, b):
    z = w1 * x1 + w2 * x2 + b
    return sigmoid(z)
```
Computes the weighted sum of inputs and passes it through the sigmoid function to obtain the predicted output.

### Loss Function
```python
def loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
```
Binary cross-entropy loss is used to measure how well the model predicts the target values.

### Gradient Computation
```python
def compute_gradients(x, y_true, y_pred):
    x1, x2 = x
    error = y_pred - y_true
    dw1 = error * x1
    dw2 = error * x2
    db = error
    return dw1, dw2, db
```
Computes the gradients of the loss with respect to weights and bias for updating the model parameters.

### Model Training
```python
model = LogisticRegression()
model.train(X, y, learning_rate=0.1, epochs=1000)
```
The model is trained using a dataset representing a NAND gate. The loss and accuracy are tracked over epochs.

### Visualization
- The training loss and accuracy are plotted over epochs.
- A decision boundary plot is generated to visualize how the model classifies inputs.

## Expected Output
- Training loss and accuracy plots.
- Model predictions on the NAND dataset.
- Decision boundary visualization.

## Testing
Several assertions are included to validate key components such as:
- Forward pass computation
- Loss function
- Gradient calculations
- Parameter updates

If all tests pass, the following message will be displayed:
```bash
All tests passed!
```

## Conclusion
This project demonstrates how logistic regression can learn a simple Boolean function like NAND. It showcases fundamental machine learning concepts such as gradient descent, loss computation, and model evaluation.

## License
This project is licensed under the MIT License.

## Author
By Mohamed Sobhy

