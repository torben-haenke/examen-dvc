import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import json

from joblib import load

X_test = pd.read_csv('data/processed_data/X_test_scaled.csv', index_col = 0)
y_test = pd.read_csv('data/processed_data/y_test.csv', index_col = 0)
y_test = y_test['silica_concentrate']

optimized_hgboost_model = load('models/optimized_hgboost_model.joblib')

y_pred = optimized_hgboost_model.predict(X_test)

# calculate some reasonable score, mse, mae, r2
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = { 'mse': mse, 'mae': mae, 'r2': r2 }

# write the metrics to the metrics folder
with open('metrics/hgboost_metrics.json', 'w') as f:
    json.dump(metrics, f)

# export the predictions to a new dataset in data/processed

df_predictions = pd.DataFrame(y_pred, columns = ['silica_concentrate'])
# get the correct index to the predictions
df_predictions.index = y_test.index

df_predictions.to_csv('data/predictions/hgboost_predictions.csv')





