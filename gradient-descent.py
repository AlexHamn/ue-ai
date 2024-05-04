import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Initialize parameters
theta = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.1
num_iterations = 1000

# Gradient descent
for _ in range(num_iterations):
    # Compute predictions
    y_pred = theta[0] + theta[1] * X
    
    # Compute gradients
    grad_0 = np.mean(y_pred - y)
    grad_1 = np.mean((y_pred - y) * X)
    
    # Update parameters
    theta[0] -= learning_rate * grad_0
    theta[1] -= learning_rate * grad_1

# Plot the results
plt.scatter(X, y)
plt.plot(X, theta[0] + theta[1] * X, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.show()