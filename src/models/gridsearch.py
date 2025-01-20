import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import json

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv', index_col = 0)
y_train = pd.read_csv('data/processed_data/y_train.csv', index_col = 0)
y_train = y_train['silica_concentrate']

# Define the model
model = HistGradientBoostingRegressor(random_state=1403)

# Define parameter grid for grid search
param_grid = {'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [None, 3, 5, 7],
                'min_samples_leaf': [10, 20, 30],
                'max_iter': [100, 200, 300]}

# Set up GridSearchCV
gridsearch = GridSearchCV(model,
                            param_grid,
                            scoring='neg_mean_squared_error',
                            cv=5,
                            verbose=2)

# Fit the model and find the best parameters
gridsearch.fit(X_train, y_train)

# Show the best parameters and score
print("Best parameters:", gridsearch.best_params_)

# Save the best parameters to a JSON file
# here I decided not to use pkl file, since I would like to able to read the output in an editor
with open('models/best_hgboost_params.json', 'w') as f:
    json.dump(gridsearch.best_params_, f, indent=4)
print("Best parameters saved to best_hgboost_params.json")