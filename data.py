import tensorflow as tf
from tensorflow import keras
import numpy as np


def load_data(dataset):
    '''
    Load dataset
    :param dataset: Dataset to load
    :param n_classes: Number of output classes
    :return: Dataset in form x_train, y_train, x_test, y_test
    '''
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        return x_train, y_train, x_test, y_test
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        return x_train, y_train, x_test, y_test
    else:
        assert False, 'Unknown dataset'


def add_gaussian_noise(data, mean=0, stddev=10, min_val=0, max_val=255):
    '''
    Add Gaussian noise to some data
    :param data: Data to add noise to
    :param mean: Mean of Gaussian noise
    :param stddev: Standard deviation of Gaussian noise
    :param min_val: Minimum value of data
    :param max_val: Maximum value of data
    :return: Data with Gaussian noise added
    '''
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, data.shape)

    # Add the noise to the data
    data_noisy = data + noise

    # Clip the data to maintain valid values
    data_noisy = np.clip(data_noisy, min_val, max_val)

    # Return noisy data
    return data_noisy


def add_label_noise(labels, noise_rate, n_classes):
    '''
    Adds mislabeling noise to data
    :param labels: Original correct labels
    :param noise_rate: Proportion of images to change labels
    :param n_classes: Number of output classes
    :return:
    '''
    # Get labels which will change
    n_samples = labels.shape[0]
    n_noisy = int(noise_rate * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    # Generate random labels, which can be any class but the original one
    for idx in noisy_indices:
        original_label = labels[idx][0]
        possible_labels = list(range(n_classes))
        possible_labels.remove(original_label)
        labels[idx] = np.random.choice(possible_labels, 1)

    # Return noisy labels
    return labels
