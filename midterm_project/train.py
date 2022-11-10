import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import pickle

# parameters
eta = 0.3
md = 3
mcw = 1
output_file = 'model.bin'


# data preparation
print('Loading the data...')
df = pd.read_csv('energy_efficiency_data.csv')


print('Preparing the data...')
del df['Cooling_Load']

df.columns = df.columns.str.lower()

numerical = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'glazing_area']
categorical = ['orientation', 'glazing_area_distribution']


print('Splitting the data...')
def split_train_val_test(df, val_size, test_size, target, random_state):
    """
    Splits the dataset into 3 parts: train/validation/test with
    (1-val_size-test_size)/val_size/test_size distribution.
    Extracts the target variable from all datasets.
    """

    # Split the dataset into 2 parts: full_train/test with (train_size+val_size)/test_size distribution
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split the full_train dataset into 2 parts: train/val with train_size/val_size distribution
    df_train, df_val = train_test_split(df_full_train, test_size=val_size / (1 - test_size), random_state=random_state)

    # Verify the shapes of datasets
    print(f'train: {df_train.shape}, val: {df_val.shape}, test: {df_test.shape}')

    # Reset indices of all datasets
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full_train = df_full_train.reset_index(drop=True)

    # Extract target variable from all datasets
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values
    y_full_train = df_full_train[target].values

    # Delete target variable from all datasets
    del df_train[target]
    del df_val[target]
    del df_test[target]
    del df_full_train[target]

    return df_full_train, y_full_train, df_train, df_val, df_test, y_train, y_val, y_test

df_full_train, y_full_train, df_train, df_val, df_test, y_train, y_val, y_test =\
            split_train_val_test(df, 0.2, 0.2, 'heating_load', 1)


# training the model
print('Implementing DictVectorizer to the data...')
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print('Creating DMatrix for XGBoost...')
features = dv.feature_names_
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': eta,
    'max_depth': md,
    'min_child_weight': mcw,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200)

# validation
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))
print(f'RMSE on validation dataset: {rmse.round(3)}')


# training the final model on full_train dataset

print('Training the final model on full_train dataset...')
print('Implementing DictVectorizer to the data...')
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

print('Creating DMatrix for XGBoost...')
features = dv.feature_names_
dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

model_final = xgb.train(xgb_params, dfull_train, num_boost_round=200)

y_pred = model_final.predict(dtest)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred)).round(3)
print(f'Final model RMSE on test dataset: {rmse_test.round(3)}')


# Save the model
print('Saving the final model with pickle...')
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_final), f_out)

print(f'The final model is saved to {output_file}.')