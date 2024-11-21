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
df_data_cb = pd.read_csv('./data/data.csv')
df_data_wos = pd.read_csv('./data/data_wos.csv')

# Preview the Clickbait data
print(f'Number of headlines in Clickbait dataset: {len(df_data_cb)}')
print('\nSample:')
x = 42
sample_cb_data = df_data_cb.iloc[x:x+5]
for _, row in sample_cb_data.iterrows():
    print(f"{row['label']}: {row['headline']}", end="")
    
print('\nClickbait dataset source format:')
print(df_data_cb[['headline', 'label']][x:x+5])

# Preview the Web of Science data
print(f'Number of Web of Science Articles: {len(df_data_wos)}')

# Numerical label to domain mapping
wos_label = {0:'CS', 1:'ECE', 4:'Civil', 5:'Medical'}

# Out labels as a table
wos_label_df = pd.DataFrame(list(wos_label.items()),columns=['Label','Description'])
print("\nKey:")
print(wos_label_df)

# Output sample labels and articles
print("\nSample:")
x=17
sample_wos_data = {
    'Label': df_data_wos['label'][x:x+5],
    'Description': [wos_label[label] for label in df_data_wos['label'][x:x+5]],
    'Article': df_data_wos['article'][x:x+5]
}

sample_wos_df = pd.DataFrame(sample_wos_data)
print(sample_wos_df)

print('\Web of Science dataset source format:')
print(df_data_wos[['article', 'label']][x:x+5])