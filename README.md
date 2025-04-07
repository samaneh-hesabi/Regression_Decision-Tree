# Diabetes Decision Tree Regression Analysis

This project implements a Decision Tree Regression model to predict diabetes progression using the scikit-learn diabetes dataset. The implementation includes model training, evaluation, and visualization components.

## Features

- Decision Tree Regression model implementation
- Comprehensive model evaluation metrics
- Cross-validation analysis
- Multiple visualization plots
- Residual analysis

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

## Code Structure

The script (`ddd.py`) performs the following operations:

1. **Data Loading and Preparation**
   - Loads the diabetes dataset from scikit-learn
   - Converts data to pandas DataFrame
   - Splits data into training and testing sets (80/20 split)

2. **Model Implementation**
   - Creates a Decision Tree Regressor with max_depth=3
   - Trains the model on the training data
   - Makes predictions on the test set

3. **Model Evaluation**
   - Calculates multiple metrics:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - R-squared (R²) score
   - Performs 5-fold cross-validation

4. **Visualization**
   - Residuals vs Predicted Values plot
   - Histogram of residuals
   - Actual vs Predicted values plot
   - Decision Tree visualization

## Output

The script generates:
- Model performance metrics
- Cross-validation results
- Four visualization plots:
  1. Residuals vs Predicted Values
  2. Histogram of Residuals
  3. Actual vs Predicted Values
  4. Decision Tree Structure

## Usage

1. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Run the script:
```bash
python ddd.py
```

## Model Performance

The model provides the following metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared) score
- Cross-validation RMSE scores and mean

## Visualization

The script generates four plots to help understand the model's performance:
1. **Residuals vs Predicted**: Shows the distribution of prediction errors
2. **Histogram of Residuals**: Displays the frequency distribution of errors
3. **Actual vs Predicted**: Compares model predictions with actual values
4. **Decision Tree**: Visualizes the structure of the decision tree model

## Notes

- The model uses a maximum tree depth of 3 to prevent overfitting
- Random state is set to 42 for reproducibility
- The dataset is automatically split into 80% training and 20% testing data
