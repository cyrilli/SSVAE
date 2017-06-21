import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR10(ROOT):
  """ load all of cifar """
  # Xtr_path = ROOT+'train_2d_unnormalized.npy'
  # Xte_path = ROOT +'test_2d_unnormalized.npy'
  Xtr_path = ROOT+'train_2d_norm_fft.npy'
  Xte_path = ROOT +'test_2d_norm_fft.npy'
  Ytr_path = ROOT + 'labels_2d.npy'
  Yte_path = ROOT + 'test_labels_normalized_no_noise_2d.npy'
  Xtr = np.load(Xtr_path).reshape(39600, 32, 32, 1)
  Ytr = np.load(Ytr_path).reshape(-1)-1
  Xte = np.load(Xte_path).reshape(3750, 32, 32, 1)
  Yte = np.load(Yte_path).reshape(-1)-1

  return Xtr, Ytr, Xte, Yte