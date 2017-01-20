import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

"""
Contains utility functions for files feature_extraction_pca.py and feature_extraction_lda.py
"""

def read_data(fin_data_name, fin_labels_name):
    """
    A utility function to read data and identify the threshold for the classifier given the file paths to the train and test data
    """
    lines1 = open(fin_data_name, 'r').readlines()
    lines2 = open(fin_labels_name, 'r').readlines()
    X, Y = [], []
    for l in lines1:
        X.append(map(float, l.split(',')))
    Y = map(lambda x: float(x.strip()), lines2)
    thresh = np.mean(list(set(Y)))
    Y = np.array(Y).reshape((-1, 1))
    return X, Y, thresh

def plot(X, X_ex, Y, model_reduced, title):
    """
    A utility function to plot the given 3-D dataset and the reduced dataset obtained by extracting features ( a single feature )
    """
    fig = plt.figure('Raw data', figsize = (4, 3))
    plt.clf()
    ax = Axes3D(fig, rect = [0, 0, .95, 1], elev = 48, azim = 134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = Y, cmap = plt.cm.binary_r, label = ['Class-1', 'Class-2'])
    x_surf = [X[:, 0].min(), X[:, 0].max(), X[:, 0].min(), X[:, 0].max()]
    y_surf = [X[:, 0].max(), X[:, 0].max(), X[:, 0].min(), X[:, 0].min()]
    x_surf = np.array(x_surf)
    y_surf = np.array(y_surf)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    fig2 = plt.figure('Projected data')
    labels = np.unique(Y.tolist())
    idx_ro = (Y == labels[0])
    idx_go = (Y == labels[1])
    p1 = plt.plot(X_ex[idx_ro], Y[idx_ro], 'ko')
    p2 = plt.plot(X_ex[idx_go], Y[idx_go], 'wo')

    c = 0.5*(labels[0] + labels[1])
    m = model_reduced.coef_[0]
    x_val = np.linspace(X_ex.min(), X_ex.max(), 11)
    y_val = c + m*x_val
    p3 = plt.plot(x_val, y_val, color = 'blue', linewidth = 2)
    plt.legend(['Class-1', 'Class-2', 'Classifier boundary'])
    plt.xlim(X_ex.min()*0.9, X_ex.max()*1.1)
    plt.ylim(Y.min()*0.9, Y.max()*1.1)
    plt.show()
