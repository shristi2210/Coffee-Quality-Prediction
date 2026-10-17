# Arabica Coffee Quality Prediction Using Linear Regression

## Overview
This project investigates whether the aroma score of Arabica coffee can be used to predict its flavor score using simple linear regression. It demonstrates the implementation of a regression model, manual calculation of evaluation metrics, validation using scikit-learn, and 5-fold cross-validation.

## Problem Statement
Can the aroma score of Arabica coffee predict its flavor score?

## Objectives
- Explore the relationship between aroma and flavor scores.
- Build a simple linear regression model.
- Calculate the regression coefficient and intercept.
- Predict flavor for a given aroma score.
- Evaluate predictions using MAE, MSE, RMSE, and R-squared.
- Compare manually calculated metrics with scikit-learn.
- Assess model performance using 5-fold cross-validation.

## Dataset
The project uses a cleaned Arabica coffee quality dataset stored in CSV format.

- Input feature (X): Aroma score
- Target variable (y): Flavor score

The scripts select these variables from the CSV by column index. Ensure the dataset schema and column order match the code before running it.

## Methodology
1. Load the CSV data.
2. Extract aroma and flavor scores.
3. Train a simple linear regression model.
4. Calculate the coefficient and intercept.
5. Predict flavor for an aroma score of 8.0.
6. Evaluate model predictions using MAE, MSE, RMSE, and R-squared.
7. Calculate metrics manually and compare them with scikit-learn.
8. Apply shuffled 5-fold cross-validation with random_state=42.

## Model Equation
Flavor = 1.01216 + (0.86014 × Aroma)

The positive coefficient indicates that higher aroma scores are associated with higher predicted flavor scores in this fitted model.

## Results

### Fitted model
- Coefficient: 0.86014
- Intercept: 1.01216
- Predicted flavor for aroma = 8.0: 7.89326

### Evaluation on the fitted dataset
- MAE: 0.16698
- MSE: 0.05386
- RMSE: 0.23207
- R-squared: 0.66309

### 5-fold cross-validation
- Mean MAE: 0.16733
- Mean RMSE: 0.23115
- Mean R-squared: 0.61244

The cross-validation results provide a more informative estimate of performance across different data splits than evaluating the model on the same data used to fit it.

## Evaluation Metrics
- **MAE:** Mean absolute difference between actual and predicted flavor scores.
- **MSE:** Mean squared prediction error.
- **RMSE:** Square root of MSE, expressed in the target variable's units.
- **R-squared:** Proportion of variation in the target explained by the fitted model on the evaluated data.

## Technologies Used
- Python
- NumPy
- scikit-learn
- CSV file handling

## How to Run

1. Clone or download this repository.
2. Install the dependencies:
   pip install -r requirements.txt
3. Ensure the CSV file is named arabica_data_cleaned.csv and is in the same directory as the scripts.
4. Run a script:
   python model.py

Other scripts calculate predictions, regression coefficients, evaluation metrics, and cross-validation results.

## Limitations
- The model uses only one input feature: aroma.
- A relationship between aroma and flavor does not establish causation.
- The reported fitted-data metrics are not independent test-set results.
- The results may not generalize to other coffee varieties or datasets.
- The scripts depend on specific CSV column indices and numeric values.

## Future Improvements
- Explore additional coffee quality attributes.
- Visualize correlations and regression results.
- Compare linear regression with other regression algorithms.
- Evaluate the model using a separate test set.
- Improve preprocessing and missing-value handling.

## Author
Shristi Sarkar

## License
Add a license if you intend to distribute the project under specific reuse terms.
