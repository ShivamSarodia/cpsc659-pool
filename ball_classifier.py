import os
import sys
import math
import cv2
from PIL import Image
import numpy as np

def split_train_test(input, target, train_percent=0.8):
    indices = range(input.shape[0])
    np.random.shuffle(indices)
    num_train = int(input.shape[0]*train_percent)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    return input[train_indices, :], target[train_indices,:], input[test_indices,:], target[test_indices,:]

def main():
    labels = np.load('train_augmented_labels.npy')
    for str_label, label in labels:
        im_path = 'train_augmented/' + str_label + '.png'

if __name__ == '__main__':
    main()
