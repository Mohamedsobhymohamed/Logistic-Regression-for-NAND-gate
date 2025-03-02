import numpy as np
import math
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass: Compute y_pred
def forward_pass(x1, x2, w1, w2, b):
    z = w1 * x1 + w2 * x2 + b
    return sigmoid(z)

# Binary Cross-Entropy Loss
def loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Avoid log(0) error
    return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Compute gradients for weight updates
def compute_gradients(x, y_true, y_pred):
    x1, x2 = x  # Unpack features
    error = y_pred - y_true
    dw1 = error * x1
    dw2 = error * x2
    db = error  # Bias gradient
    return dw1, dw2, db

# Update parameters using gradient descent
def update_parameters(w1, w2, b, dw1, dw2, db, learning_rate):
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db
    return w1, w2, b

# Logistic Regression Model
class LogisticRegression:
    def __init__(self):
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.b = np.random.randn()
        self.losses = []
        self.accuracies = []

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                
                # Forward pass
                y_pred = forward_pass(x[0], x[1], self.w1, self.w2, self.b)
                
                # Compute loss
                epoch_loss += loss(y_true, y_pred)
                
                # Compute gradients
                dw1, dw2, db = compute_gradients(x, y_true, y_pred)
                
                # Update parameters
                self.w1, self.w2, self.b = update_parameters(self.w1, self.w2, self.b, dw1, dw2, db, learning_rate)
                
                # Compute accuracy
                y_pred_label = 1 if y_pred >= 0.5 else 0
                if y_pred_label == y_true:
                    correct += 1
            
            # Track loss and accuracy
            accuracy = correct / len(X)
            self.losses.append(epoch_loss / len(X))  # Average loss per epoch
            self.accuracies.append(accuracy)
            
            # Print training progress
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")

    def predict(self, X):
        predictions = []
        for x in X:
            y_pred = forward_pass(x[0], x[1], self.w1, self.w2, self.b)
            predictions.append(1 if y_pred >= 0.5 else 0)
        return predictions

# NAND Gate Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, 1, 0])  # NAND truth table

# Train Model
model = LogisticRegression()
model.train(X, y, learning_rate=0.1, epochs=1000)

# Plot Training Loss
plt.plot(model.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Plot Training Accuracy
plt.plot(model.accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.show()

# Predictions
predictions = model.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {predictions[i]}")

# Decision Boundary Plot
x1 = np.linspace(-0.5, 1.5, 100)
x2 = np.linspace(-0.5, 1.5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[forward_pass(a, b, model.w1, model.w2, model.b) for a, b in zip(X1_row, X2_row)] for X1_row, X2_row in zip(X1, X2)])

plt.contourf(X1, X2, Z, levels=1, colors=['blue', 'red'], alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary')
plt.show()

# Test Forward Pass
assert math.isclose(forward_pass(0, 0, 1, 1, 0.5), 0.6224593312018546, rel_tol=1e-9)

# Test Loss Function
y_true = 0
y_pred = 0.6224593312018546
assert math.isclose(loss(y_true, y_pred), 0.9740769841801068, rel_tol=1e-9)

# Test Gradient Computation
x = [0, 0]
dw1, dw2, db = compute_gradients(x, y_true, y_pred)
assert math.isclose(dw1, 0, rel_tol=1e-9)
assert math.isclose(dw2, 0, rel_tol=1e-9)
assert math.isclose(db, 0.6224593312018546, rel_tol=1e-9)

# Test Parameter Update
w1, w2, b = 1.0, 1.0, 0.5
dw1, dw2, db = 0.1, -0.2, 0.05
learning_rate = 0.01
w1, w2, b = update_parameters(w1, w2, b, dw1, dw2, db, learning_rate)
assert math.isclose(w1, 0.999, rel_tol=1e-9)
assert math.isclose(w2, 1.002, rel_tol=1e-9)
assert math.isclose(b, 0.4995, rel_tol=1e-9)

print("All tests passed! ")
