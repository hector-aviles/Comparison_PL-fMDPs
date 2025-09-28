# Re-run the linear regression slope calculation since execution state was reset

import numpy as np
from sklearn.linear_model import LinearRegression

# Data points for each model (1%, 50%, 100% training data)
percentages = np.array([1, 50, 100]).reshape(-1, 1)

# Corresponding training times
training_times = {
    "PL-fMDPs + APs": [3.77, 8.32, 10.56],
    "CART": [0.234, 7.2, 18],
    "MLP": [86.9, 618.56, 918.4],
    "LR": [0.09, 6.6, 14.4],
    "NB": [0.006, 0.24, 0.6],
    "RF": [4.32, 120.6, 259.2],
    "XGBoost": [7.2, 68.4, 135],
}

# Compute slopes using linear regression
slopes = {}
for model, times in training_times.items():
    reg = LinearRegression().fit(percentages, times)
    slopes[model] = reg.coef_[0]

slopes
