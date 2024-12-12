import pandas as pd
import numpy as np
import scipy as sp
import sys
import re

from copy import deepcopy
from sklearn.metrics import accuracy_score

print('Versions')
print('python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))

# Load the datasets
df_data_cb = pd.read_csv('./data/data_cb.csv')
df_data_wos = pd.read_csv('./data/web_of_science_data.csv') #('./data/data_wos.csv')

# Preview the Clickbait data
print(f'\nNumber of headlines in Clickbait dataset: {len(df_data_cb)}')
print('\nSample:')
x = 42
sample_cb_data = df_data_cb.iloc[x:x+5]
for _, row in sample_cb_data.iterrows():
    print(f"{row['label']}: {row['headline']}", end="")
    
print('\nClickbait dataset source format:')
print(df_data_cb[['headline', 'label']][x:x+5])

# Preview the Web of Science data
print(f'\nNumber of Web of Science Articles: {len(df_data_wos)}')

# Numerical label to domain mapping
wos_label = dict(zip(df_data_wos['Y1'], df_data_wos['Domain']))

# Output labels as a table
wos_label_df = pd.DataFrame(list(wos_label.items()),columns=['Label','Description'])
print("\nKey:")
print(wos_label_df)

# Output sample labels and articles
print("\nSample:")
x=17
sample_wos_data = {
    'Label': df_data_wos['Y1'][x:x+5],
    'Description': [wos_label.get(label.strip(), label) for label in df_data_wos['Domain'][x:x+5]],
    'Article': df_data_wos['Abstract'][x:x+5]
}

sample_wos_df = pd.DataFrame(sample_wos_data)
print(sample_wos_df)

print('\nWeb of Science dataset source format:')
print(df_data_wos[x:x+5])

import split

# split the data into 80/20 train/test sets
df_train_cb, df_test_cb = split.split_data(df_data_cb)
df_train_wos, df_test_wos = split.split_data(df_data_wos)

# separate the independent and dependent variables
x_train_cb, y_train_cb = list(df_train_cb['headline']), list(df_train_cb['label'])
x_test_cb, y_test_cb = list(df_test_cb['headline']), list(df_test_cb['label'])
x_train_wos, y_train_wos = list(df_train_wos['Abstract']), list(df_train_wos['Domain'])
x_test_wos, y_test_wos = list(df_test_wos['Abstract']), list(df_test_wos['Domain'])

print(f'\nClickbait dataset training headline count: {len(x_train_cb)}')
print(f'Clickbait dataset test headline count: {len(x_test_cb)}')
print(f'\nWeb of Science training article count: {len(x_train_wos)}')
print(f'Web of Science test article count: {len(x_test_wos)}')

# Clean the Clickbait headlines
from preprocess import Preprocess
cleaned_train_cb = Preprocess().clean_dataset(x_train_cb)
cleaned_test_cb = Preprocess().clean_dataset(x_test_cb)

# Clean the Web of Science articles
from preprocess import clean_wos
cleaned_train_wos, cleaned_test_wos = clean_wos(x_train_wos, x_test_wos)