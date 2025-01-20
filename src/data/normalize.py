import pandas as pd

from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('data/processed_data/X_train.csv', index_col = 0)
X_test = pd.read_csv('data/processed_data/X_test.csv', index_col = 0)

# I decided to go with the standart scaler, from quickly looking at the data
# I think most of the variables are normal distributed, but more preprocessing would be need
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)

# reset the index, in case it might be needed later, since it got lost during scaling
X_train_scaled.index = X_train.index
X_test_scaled.index = X_test.index

X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv')
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv')