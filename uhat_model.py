import numpy as np
from PIL import Image

with np.load('uhat_dataset.npz') as data:
    train_chars_data = data['x_chars_train']
    train_digits_data = data['x_digits_train']
    test_chars_data = data['x_chars_test']
    test_digits_data = data['x_digits_test']
    train_chars_labels = data['y_chars_train']
    train_digits_labels = data['y_digits_train']
    test_chars_labels = data['y_chars_test']
    test_digits_labels = data['y_digits_test']

# shift values of character labels to avoid overlapping with digit labels
train_chars_labels = [(label_val + 10) for label_val in train_chars_labels]
test_chars_labels = [(label_val + 10) for label_val in test_chars_labels]

# merge character and digit data
train_data = np.concatenate([train_chars_data, train_digits_data], axis=0)
test_data = np.concatenate([test_chars_data, test_digits_data], axis=0)
train_labels = np.concatenate([train_chars_labels, train_digits_labels])
test_labels = np.concatenate([test_chars_labels, test_digits_labels])

