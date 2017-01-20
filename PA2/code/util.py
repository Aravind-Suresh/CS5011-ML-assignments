"""
Consists of utility functions for svm_classifier.py & neural_network.py
"""
# Importing modules
import sys,os
import numpy as np
import csv

def read_data(path):
    """
    A utility function to read data given path to the input file
    """
    arr = []
    with open(path, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            arr.append(map(float, row))
    return np.array(arr)

def write_data(data, path):
    """
    A utility function to write data given path to the output file
    """
    with open(path, 'wb') as file:
        writer = csv.writer(file, delimiter=',')
        for x,y in data:
            writer.writerow(map(str, x.tolist() + [y]))

def one_hot(idx, l):
    """
    A utility function to encode a label in the one-hot format,
    given the total length of the vector and the label.
    """
    ret = [0]*l
    ret[idx] = 1
    return ret
