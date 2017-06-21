from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
import collections

import numpy as np

import time

def make_bearing_dataset(data_dir,n_validation = 0,vectorize=False, num_labeled_samples = 1320):
    NUM_CLASSES = 10

    X_train, y_train, X_test, y_test = load_CIFAR10(data_dir)
    # 39600, 32, 32, 1   39600,
    #  3750, 32, 32, 1   3750,
    
    NUM_TRAIN   = X_train.shape[0]
    NUM_TEST    = X_test.shape[0]
    # reshape to vectors
    if vectorize:
        X_train = np.reshape(X_train,(X_train.shape[0],-1))  # 39600, 1024
        X_test  = np.reshape(X_test,(X_test.shape[0],-1))    # 3750, 1024

    # make one-hot coding
    y_train_temp = np.zeros((NUM_TRAIN,NUM_CLASSES))
    for i in range(NUM_TRAIN):
        y_train_temp[i,y_train[i]] = 1
    y_train = y_train_temp   # 39600, 10

    y_test_temp = np.zeros((NUM_TEST,NUM_CLASSES))
    for i in range(NUM_TEST):
        y_test_temp[i,y_test[i]] = 1
    y_test = y_test_temp    # 3750, 10

    X_train_labeled, y_train_labled = draw_labeled_data(X_train, y_train, labeled_sample_per_category= num_labeled_samples)

    return (X_train, y_train, X_train_labeled, y_train_labled, X_test, y_test)

def read_bearing_dataset(data_dir,n_validation = 0,vectorize=False, num_labeled_samples = 1320):
    X_train_unlab, y_train_unlab, X_train_lab, y_train_lab, X_test, y_test = make_bearing_dataset(data_dir,n_validation,vectorize, num_labeled_samples)
    Dataset = {
        'train_data_unlabeled' : X_train_unlab,
        'train_label_unlabeled': y_train_unlab,
        'train_data_labeled': X_train_lab,
        'train_label_labeled': y_train_lab,
        'test_data': X_test,
        'test_label': y_test
    }
    return Dataset

# to be used in ipython notebook
def visualize_dataset(dataset,height=0,width=0,channels=0):
    images = dataset.images
    labels = dataset.labels
    num_classes = labels.shape[1]
    samples_per_class = 7

    for cls in range(num_classes):
        idxs = np.flatnonzero(labels[:,cls] == 1)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + cls + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            if channels == 1:
                plt.imshow(images[idx].reshape((height,width)))
            elif channels > 1:
                plt.imshow(images[idx].reshape((height,width,channels)))
            else:
                plt.imshow(images[idx])
            plt.axis('off')
            if i == 0:
                plt.title('C{}'.format(cls))
    plt.show()

def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string

def draw_labeled_data(data, label, component_size = 1320, labeled_sample_per_category = 1320):
    '''
    This function aims at reducing data size while making sure that the distribution of data is not changed
    :param data: shape (39600, 32, 32, 1)
                In the data, three different load conditions, each 13200
                In each load condition, ten different fault types, each 1320
                To reduce the data size, we need to sample from each 1320-size component
    :param label: shape (39600, )
    :param component_size: an integer represents the minimum size of a component of the data 
    :param keep_ratio: the percentage of data that we wish to keep
    :return: shape (39600*keep_ratio, 32, 32, 256)
    '''
    assert labeled_sample_per_category <= 1320  # 30 categories in total
    
    data_length = data.shape[0]
    indices = np.arange(component_size)  # (1320,)
    np.random.shuffle(indices)  # shuffle order
    excerpt_component = indices[0:int(labeled_sample_per_category)]  # (1320*keep_ratio, ) indices of samples to be chosen from a component
    excerpt_data= []
    for start_idx in range(0, data_length - component_size + 1, component_size):
        excerpt_data.append(excerpt_component+start_idx)
    excerpt_data = np.hstack(excerpt_data)

    return data[excerpt_data, :], label[excerpt_data]

def data_generator(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    start_idx = 0
    while True:
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        start_idx = start_idx + batch_size
        if start_idx > len(inputs) - batch_size:
            start_idx = 0
        yield inputs[excerpt], targets[excerpt]