import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Read CSV file
with open("arabica_data_cleaned.csv", "r", encoding="utf-8") as file:
    data = list(csv.reader(file))

# X = Aroma
X = [[float(row[20])] for row in data[1:]]

# y = Flavor
y = [float(row[21]) for row in data[1:]]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict Flavor values
y_pred = model.predict(X)

# MAE using sklearn
mae = mean_absolute_error(y, y_pred)

print("Scikit-learn MAE:", mae)