# Created using the following tutorial: https://www.tensorflow.org/tutorials/load_data/csv

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import pandas as pd
import numpy as np
import tensorflow as tf


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'survived'
LABELS = [0, 1]


def print_pandas_df(file_path):
    df = pd.read_csv(file_path)
    # df.head()
    print(df)


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


print('--------------')
print('Loading data:')
print('--------------')
print(print_pandas_df(test_file_path))


SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
raw_train_data = get_dataset(train_file_path,
                             select_columns=SELECT_COLUMNS,
                             column_defaults=DEFAULTS)
raw_test_data = get_dataset(test_file_path)

print('--------------')
print(raw_test_data)


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


print('--------------')
print('Showing batch:')
print('--------------')
show_batch(raw_train_data)


print('--------------')
print('Packed data:')
print('--------------')

packed_dataset = raw_train_data.map(pack)

for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())


def get_categorical_data():
    CATEGORIES = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
        }

    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    return categorical_columns


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']
packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data-mean)/std


def get_numeric_data():
    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]
    return numeric_columns


preprocessing_layer = tf.keras.layers.DenseFeatures(
        get_categorical_data()
        + get_numeric_data())


model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metris=['accuracy'])


train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=20)
