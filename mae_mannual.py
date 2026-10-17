import csv
import numpy as np

# Read CSV file
with open("arabica_data_cleaned.csv", "r", encoding="utf-8") as file:
    data = list(csv.reader(file))

# Actual Flavor values
y = np.array([float(row[21]) for row in data[1:]])

# Aroma values
X = np.array([float(row[20]) for row in data[1:]])

# Calculate mean
x_mean = np.mean(X)
y_mean = np.mean(y)

# Calculate slope
m = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)

# Calculate intercept
c = y_mean - m * x_mean

# Predicted values
y_pred = m * X + c

# Manual MAE
mae = np.sum(np.abs(y - y_pred)) / len(y)

print("Manual MAE:", mae)