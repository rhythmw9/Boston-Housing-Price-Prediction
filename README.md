# Boston-Housing-Price-Prediction

### Project Overview
This project uses regression analysis to predict housing prices in the Boston area, based on historical data from 1978. The objective is to build and compare two regression models—least squares regression and L1 regression—using polynomial features.

### Problem Description
This project aims to predict housing prices in the Boston area using data from 1978, containing 506 observations across 13 variables. The first 12 variables are features that describe various socio-economic and environmental aspects of the neighborhoods, while the 13th variable is the median housing price (in $1000’s) we aim to predict. The features are as follows:

1. Per capita crime rate by town
2. Proportion of residential land zoned for large lots
3. Proportion of non-retail business acres per town
4. Charles River dummy variable (1 if tract bounds river, 0 otherwise)
5. Nitric oxide concentration (parts per 10 million)
6. Average number of rooms per dwelling
7. Proportion of owner-occupied units built prior to 1940
8. Weighted distances to Boston employment centers
9. Index of accessibility to radial highways
10. Property tax rate per $10,000
11. Pupil-teacher ratio by town
12. Percentage of the population with lower economic status
13. Median housing price (target variable)

We implement two regression models to predict housing prices based on these features:

Least Squares Regression: Minimizes the Mean Squared Error (MSE) between predicted and actual housing prices, using linear and polynomial models.

L1-Norm Regression: Minimizes the sum of absolute deviations, making it more robust to outliers in the data.

To evaluate model performance, the data is divided into training and test sets, with training set sizes of 30% and 60%. The models are trained on each subset, and we report the Mean Squared Error for both training and test sets. Additionally, we explore the impact of data scaling and different data splits on model accuracy.


### Project Structure
- **src/**: Contains code for data preprocessing, model training, and evaluation.
- **data/**: Dataset (if available).
- **results/**: Saved results, visualizations, or metrics for model performance.

### Models Implemented
- **Least Squares Regression**: Optimized for Mean Squared Error (MSE).
- **L1 Regression**: Optimized using the L1 norm for robustness against outliers.


### Results
This project evaluated model performance by comparing Mean Absolute Error (MAE) for different regression techniques and training set sizes. The table below summarizes the results for both Linear Regression (Least Squares) and L1 Regression (Lasso):

Model	                 Training Set Size	   Train MAE	    Test MAE
Linear Regression	        30%	                3.58	         3.58
Linear Regression	        60%	                3.20           3.55
L1 Regression (Lasso)	    30%	                3.87	         3.73
L1 Regression (Lasso)	    60%	                3.91	         3.74

Key Observations
-Linear Regression: The train and test errors are close across both training set sizes, suggesting that the model is not overfitting and has learned general patterns in the data.
-L1 Regression (Lasso): While slightly higher in MAE, L1 regression is more robust to outliers, which can make it more reliable for data with potential anomalies.
-Training Set Size Effect: Increasing the training set size from 30% to 60% generally decreases the train error slightly while keeping test error stable. This indicates that the models benefit from additional data without sacrificing generalization.


### Future Work
Potential improvements include additional data preprocessing, feature engineering, and experimenting with higher-degree polynomial models.

### Contact
For any questions, feel free to reach out or connect with me on [LinkedIn] (https://www.linkedin.com/in/rhythm-winicour-freeman-975b74289/).
