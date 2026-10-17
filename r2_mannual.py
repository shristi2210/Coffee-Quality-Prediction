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

# Calculate predicted values
y_pred = m * X + c

# Calculate R²
ss_total = np.sum((y - y_mean) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)

r2 = 1 - (ss_residual / ss_total)

print("Manual R²:", r2)