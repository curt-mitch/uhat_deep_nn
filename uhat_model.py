import numpy as np
from PIL import Image

with np.load('UHaT/uhat_dataset.npz') as data:
    train_chars = data['x_chars_train']
    train_digits = data['x_digits_train']
    test_chars = data['x_chars_test']
    test_digits = data['x_digits_test']
    train_chars_labels = data['y_chars_train']
    train_digits_labels = data['y_digits_train']
    test_chars_labels = data['y_chars_test']
    test_digits_labels = data['y_digits_test']
