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


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

print('--------------')
print(raw_test_data)


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


print('--------------')
print('Showing batch:')
print('--------------')
show_batch(raw_train_data)
