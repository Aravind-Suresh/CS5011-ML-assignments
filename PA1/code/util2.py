import sys,os
import numpy as np

"""
Contains utility functions for data_imputation.py, linear_regression.py and regularized_linear_regression.py
"""

def read_data(fin_name):
    """
    A utility function to read data from the file, given its path
    """
    fin = open(fin_name,'r')
    lines = fin.readlines()
    arr=[]
    for l in lines:
        ll = l.strip()
        a = ll.split(',')
        arr.append(a)
    fin.close()
    return arr

def write_data(fout_name, data_list):
    """
    A utility function to write data to the file, given its path
    """
    fout = open(fout_name,'w')
    for i in range(len(data_list)):
        fout.write(','.join(data_list[i]))
        fout.write('\n')
    fout.close()
