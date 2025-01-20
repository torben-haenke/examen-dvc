import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw_data/raw.csv')
# the date is not a describing variable and silica_concentrate is the target variable
df = df.drop(columns = ['date'])

df_train, df_test = train_test_split(df, test_size = 0.25, random_state = 1403)

X_train = df_train.drop(columns = ['silica_concentrate'])
y_train = df_train['silica_concentrate']

X_test = df_test.drop(columns = ['silica_concentrate'])
y_test = df_test['silica_concentrate']

X_train.to_csv('data/processed_data/X_train.csv')
X_test.to_csv('data/processed_data/X_test.csv')
y_train.to_csv('data/processed_data/y_train.csv')
y_test.to_csv('data/processed_data/y_test.csv')