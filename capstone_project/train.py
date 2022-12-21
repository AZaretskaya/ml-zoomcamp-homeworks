import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle

# parameters
C = 1
max_iter = 2000
penalty = 'l1'
random_state = 1
solver = 'liblinear'
output_file = 'model.bin'


# data preparation
print('Loading the data...')
df = pd.read_csv('advertising.csv')

print('Preparing the data...')
df.drop(['Timestamp'], inplace=True, axis=1)
df.drop(['Ad Topic Line'], inplace=True, axis=1)

df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.rename(columns={'male':'gender'})
df.gender = df.gender.map({1: 'Male', 0: 'Female'}) 


categorical = df.select_dtypes(include=['object']).columns.tolist()
numerical = df.select_dtypes(exclude=['object']).columns.tolist()


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
    df_train, df_val = train_test_split(df_full_train, test_size=val_size/(1-test_size), random_state=random_state)
    
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
            split_train_val_test(df, 0.2, 0.2, 'clicked_on_ad', 1)


# training the model
print('Implementing DictVectorizer to the data...')
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=random_state)
model.fit(X_train, y_train)

# validation
y_pred = model.predict(X_val)
print(f'AUC on validation dataset: {roc_auc_score(y_val, y_pred).round(3)}.')

# training the final model on full_train dataset

print('Training the final model on full_train dataset...')
print('Implementing DictVectorizer to the data...')
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

model_final = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=random_state)
model_final.fit(X_full_train, y_full_train)

y_pred = model_final.predict(X_test)
print(f'Final model AUC on test dataset: {roc_auc_score(y_test, y_pred).round(3)}.')

# Save the model
print('Saving the final model with pickle...')
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_final), f_out)

print(f'The final model is saved to {output_file}.')


