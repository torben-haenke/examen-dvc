import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

import json
from joblib import dump

# this is somehow redundent, it should look like the same as in gridsearch.py
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv', index_col = 0)
y_train = pd.read_csv('data/processed_data/y_train.csv', index_col = 0)
y_train = y_train['silica_concentrate']


# read the best parameter from the gridsearch
with open('models/best_hgboost_params.json', 'r') as f:
    loaded_params = json.load(f)

# initialize a model with the best parameters from the gridsearch
# and train the model
optimized_model = HistGradientBoostingRegressor(random_state=1403, **loaded_params)
optimized_model.fit(X_train, y_train)

dump(optimized_model, 'models/optimized_hgboost_model.joblib')