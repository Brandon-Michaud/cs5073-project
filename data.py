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
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        return x_train, y_train, x_test, y_test
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        return x_train, y_train, x_test, y_test
    else:
        assert False, 'Unknown dataset'


def add_gaussian_noise(data, mean=0, stddev=10, min_val=0, max_val=255):
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, data.shape)

    # Add the noise to the data
    images_noisy = data + noise

    # Clip the data to maintain valid values
    images_noisy = np.clip(images_noisy, min_val, max_val)

    return images_noisy


def add_label_noise(labels, noise_rate, n_classes):
    n_samples = labels.shape[0]
    n_noisy = int(noise_rate * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    # Generate random labels, which can be any class but the original one
    for idx in noisy_indices:
        original_label = labels[idx][0]
        possible_labels = list(range(n_classes))
        possible_labels.remove(original_label)
        labels[idx] = np.random.choice(possible_labels, 1)

    return labels
