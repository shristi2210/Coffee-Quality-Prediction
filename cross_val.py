import csv
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression

# Read CSV file
with open("arabica_data_cleaned.csv", "r", encoding="utf-8") as file:
    data = list(csv.reader(file))

# X = Aroma
X = np.array([[float(row[20])] for row in data[1:]])

# y = Flavor
y = np.array([float(row[21]) for row in data[1:]])

# K-Fold Cross Validation
RANDOM_STATE = 42

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Create Linear Regression model
model = LinearRegression()

# Define evaluation metrics
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

# Perform Cross Validation
results = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring=scoring
)

# Convert negative errors to positive
rmse = -results["test_rmse"].mean()
mae = -results["test_mae"].mean()
r2 = results["test_r2"].mean()

print("\nCross-Validation Results")
print("------------------------")
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)