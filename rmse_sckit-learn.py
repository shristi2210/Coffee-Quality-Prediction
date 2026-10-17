import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# Read CSV file
with open("arabica_data_cleaned.csv", "r", encoding="utf-8") as file:
    data = list(csv.reader(file))

# X = Aroma
X = [[float(row[20])] for row in data[1:]]

# y = Actual Flavor
y = [float(row[21]) for row in data[1:]]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predicted values
y_pred = model.predict(X)

# RMSE using sklearn
rmse = root_mean_squared_error(y, y_pred)

print("Scikit-learn RMSE:", rmse)