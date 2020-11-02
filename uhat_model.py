import numpy as np
# from PIL import Image

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
train_data = np.concatenate([train_digits_data, train_chars_data], axis=0)
test_data = np.concatenate([test_digits_data, test_chars_data], axis=0)
train_labels = np.concatenate([train_digits_labels, train_chars_labels])
test_labels = np.concatenate([test_digits_labels, test_chars_labels])

# reshape to flatten images
train_data = train_data.reshape(len(train_data), 28*28)
test_data = test_data.reshape(len(test_data), 28*28)

# create one-hot encodings for labels
one_hot_labels = np.zeros((len(train_labels), 50))

for i, l in enumerate(train_labels):
    one_hot_labels[i][l] = 1
train_labels = one_hot_labels

# set up loss function and hyperparameters
np.random.seed(1)


def relu(x):
    '''return x if x > 0, otherwise return 0'''
    return (x >= 0) * x


def relu2deriv(x):
    '''derivative of relu: return 1 if x > 0, otherwise return 0'''
    return x >= 0


alpha = 0.005
iterations = 350
hidden_size = 40
pixels_per_image = 28*28
num_labels = 50
weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1
