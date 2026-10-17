from sklearn.linear_model import LinearRegression
import csv

with open("arabica_data_cleaned.csv", "r", encoding="utf-8") as file:
    data = list(csv.reader(file))

# X = Aroma
X = [[float(row[20])] for row in data[1:]]

# y = Flavor
y = [float(row[21]) for row in data[1:]]

model = LinearRegression()
model.fit(X, y)

predicted_flavor = model.predict([[8.0]])

print("Predicted Flavor:", predicted_flavor)