import sys
import numpy as np
# from PIL import Image

with np.load('uhat_dataset.npz') as data:
    train_data = data['x_chars_train']
    test_data = data['x_chars_test']
    train_labels = data['y_chars_train']
    test_labels = data['y_chars_test']

# reshape to flatten images
train_data = train_data.reshape(len(train_data), 28*28) / 255
test_data = test_data.reshape(len(test_data), 28*28) / 255

# create one-hot encodings for labels
one_hot_train_labels = np.zeros((len(train_labels), 40))
one_hot_test_labels = np.zeros((len(test_labels), 40))

for i, l in enumerate(train_labels):
    one_hot_train_labels[i][l] = 1
train_labels = one_hot_train_labels

for i, l in enumerate(test_labels):
    one_hot_test_labels[i][l] = 1
test_labels = one_hot_test_labels

# set up loss function and hyperparameters
np.random.seed(1)

# TODO: https://stackoverflow.com/questions/7559595/python-runtimewarning-overflow-encountered-in-long-scalars


def relu(x):
    '''return x if x > 0, otherwise return 0'''
    return (x >= 0) * x


def relu2deriv(x):
    '''derivative of relu: return 1 if x > 0, otherwise return 0'''
    return x >= 0


alpha = 0.005
# iterations = 350
iterations = 1
hidden_size = 40
pixels_per_image = 28*28
num_labels = 40
weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error = 0.0
    correct_count = 0
    for i in range(1000):
        layer_0 = train_data[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        error += np.sum((train_labels[i:i+1] - layer_2) ** 2)
        correct_count += int(np.argmax(layer_2) ==
                             np.argmax(train_labels[i:i+1]))
        layer_2_delta = (train_labels[i:i+1] - layer_2)
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T)\
            * relu2deriv(layer_1)
        weights_1_2 += alpha * np.dot(layer_1.T, layer_2_delta)
        weights_0_1 += alpha * np.dot(layer_0.T, layer_1_delta)
        if i % 1000 == 0:
            print('weights', weights_0_1)
    sys.stdout.write("\r" +
                     " I:" + str(j) +
                     " Error:" + str(error / float(len(train_data)))[0:5] +
                     " Correct:" + str(correct_count / float(len(train_data))))
