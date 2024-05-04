import matplotlib.pyplot as plt

# Actual values
y_true = [3, -0.5, 2, 7]

# Predicted values
y_pred = [2.5, 0.0, 2, 8]

# Indices for plotting
indices = range(len(y_true))

# Plot the actual and predicted values
plt.plot(indices, y_true, 'ro-', label='True Values')
plt.plot(indices, y_pred, 'b^-', label='Predicted Values')
plt.legend()
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.show()