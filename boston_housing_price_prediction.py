import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
data = np.loadtxt('housing.txt')
X = data[:, :-1]
y = data[:, -1]

# Rescale features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Define training set sizes
train_sizes = [0.3, 0.6]

# Model 1: Linear Regression
for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate MSE
    train_mse = mean_absolute_error(y_train, y_train_pred)
    test_mse = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Linear Regression with {int(size*100)}% training data - Train MSE: {train_mse}, Test MSE: {test_mse}")

# Model 2: Polynomial Regression (Third Degree)
# (Expand X to include polynomial terms before fitting)

# Model 3: `L1` Norm Regression
for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size)
    model = Lasso(alpha=1.0)  # L1 Regularization as proxy for `L1` regression
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate MSE
    train_mse = mean_absolute_error(y_train, y_train_pred)
    test_mse = mean_absolute_error(y_test, y_test_pred)
    
    print(f"L1 Regression with {int(size*100)}% training data - Train MSE: {train_mse}, Test MSE: {test_mse}")
