from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv

with open(r"C:\Users\LENOVO\OneDrive\Desktop\coffee\arabica_data_cleaned.csv",
          "r", encoding="utf-8") as file:
    data = list(csv.reader(file))

# X = Aroma
X = [[float(row[20])] for row in data[1:]]

# y = Flavor
y = [float(row[21]) for row in data[1:]]

model = LinearRegression()
model.fit(X, y)

# Coefficient and intercept
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Prediction
print("Prediction for Aroma 8.0:", model.predict([[8.0]]))

# Scatter plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Aroma")
plt.ylabel("Flavor")
plt.title("Linear Regression - Coffee Quality")
plt.show()