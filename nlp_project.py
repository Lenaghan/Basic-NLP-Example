import pandas as pd
import numpy as np
import scipy as sp
import sys
import re
import time

from copy import deepcopy
from sklearn.metrics import accuracy_score

print('Versions')
print('python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))

# Load the datasets
df_data_cb = pd.read_csv('./data/clickbait_data.csv')
df_data_wos = pd.read_csv('./data/web_of_science_data.csv')

# Preview the Clickbait data
print(f'\nNumber of Clickbait headlines: {len(df_data_cb)}')
x = 42
print('Source preview:')
print(df_data_cb[x:x+5])

# Preview the Web of Science data
print(f'\nNumber of Web of Science Articles: {len(df_data_wos)}')

#from preprocess import process_key_column
df_data_wos['Domain'] = df_data_wos['Domain'].str.strip()
df_data_wos['Domain'] = df_data_wos['Domain'].str[0].str.upper() + df_data_wos['Domain'].str[1:]

# Numerical label to domain mapping
wos_label = dict(zip(df_data_wos['Y1'], df_data_wos['Domain']))

# Output sample labels and articles
x=17
print('Source preview:')
print(df_data_wos[x:x+5])

# Output domain labels as a table
wos_label_df = pd.DataFrame(list(wos_label.items()),columns=['Label','Description'])
print("\nKey:")
print(wos_label_df)

# split the data into 80/20 train/test sets
import split
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
print(f'Web of Science test article count: {len(x_test_wos)}\n')

# Clean the Clickbait headlines
from preprocess import Preprocess
cleaned_train_cb = Preprocess().clean_dataset(x_train_cb)
cleaned_test_cb = Preprocess().clean_dataset(x_test_cb)

# Clean the Web of Science articles
from preprocess import clean_wos
cleaned_train_wos, cleaned_test_wos = clean_wos(x_train_wos, x_test_wos)

# Convert cleaned text to a Bag of Words representation
from bow_onehot_encoder import bow_encode_with_onehot
from bow_cv_encoder import bow_encode_with_cv

# Convert using OneHotEncoder
# bow_encode_with_onehot(cleaned_train_cb, cleaned_test_cb, 'cb_train_ohe_bow.pkl', 'cb_test_ohe_bow.pkl')
# bow_encode_with_onehot(cleaned_train_wos, cleaned_test_wos, 'wos_train_ohe_bow.pkl', 'wos_test_ohe_bow.pkl')

# Convert using CountVectorizer
# bow_encode_with_cv(cleaned_train_cb, cleaned_test_cb, 'cb_train_cv_bow.pkl', 'cb_test_cv_bow.pkl')
# bow_encode_with_cv(cleaned_train_wos, cleaned_test_wos, 'wos_train_cv_bow.pkl', 'wos_test_cv_bow.pkl')

# Convert to TF-IDF weighted features
from tf_idf_transformer import fit_and_transform
fit_and_transform(cleaned_train_cb, cleaned_test_cb, 'cb_train_tfidf.pkl', 'cb_test_tfidf.pkl')
fit_and_transform(cleaned_train_wos, cleaned_test_wos, 'wos_train_tfidf.pkl', 'wos_test_tfidf.pkl')