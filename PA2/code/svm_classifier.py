"""
Program to demonstrate training SVM on the features extracted from images ( DS2 )
Sample run:
    $ python svm_classifier.py ../dataset/DS2_train.csv ../dataset/DS2_test.csv ../models

NOTE:
    The models are provided as .pkl files. These should be run on the DS2_{train,test}_norm.csv files.
This is because, in the python script I am normalizing the inputs and saving it to *_norm.csv files.
So, if the model is used separately by a third-party program it should be on the normalized dataset.

Also, to import the models and test code, use:

    model = joblib.load('model1.pkl')
    # Use model.predict to predict
"""

# Importing modules
import sys,os
import numpy as np
import csv

# Importing utility functions
from util import *

# Importing SVM-related functions
from sklearn import svm
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

def get_accuracy(p, a):
    """
    A utility function to calculate accuracy given the predicted & actual values ( in % )
    """
    return 100.*np.sum(p==a)/p.shape[0]

# NOTE
# Kernel parameters for SVM | (.) - C-SVM parameters
# linear    : (C)
# poly      : coef0, degree, (C)
# rbf       : gamma, (C)
# sigmoid   : coef0, (C)

def analyze():
    """
    Function to analyze different kernel parameters and calculate the optimal values for the parameters,
    for the best fit.
    """
    # Range for C ( C SVMs )
    C_range = np.logspace(-4, 4, 10)
    # Range for gamma - the free parameter in rbf kernel
    gamma_range = np.logspace(-4, 4, 10)
    # Range for coef0 - the independent parameter in poly/sigmoid kernels
    coef0_range = np.linspace(-0.5, 0.5, 11)
    # Range for degree - Degree of the poly kernel
    degree_range = range(2, 6)

    # A container for storing the results
    res = {}

    # Analyzing all possible values of C for linear kernel
    kernel_type = 'linear'
    d={}
    d['kernel'] = kernel_type
    d['max_iter'] = 100000
    print '\nKernel:', kernel_type
    print ""
    r = {}
    for C in C_range:
        d['C'] = C
        acc = []
        # Performing K-fold cross validation test
        for train, test in k_fold:
            model = OneVsRestClassifier(svm.SVC(**d))
            model.fit(train_X[train], train_Y[train])
            pred = model.predict(train_X[test])
            acc.append(get_accuracy(pred, train_Y[test]))
        r[(C,)] = list(acc)
        print 'C =',C,'Avg.acc =',np.mean(acc)
    res[kernel_type] = r.copy()

    # Analyzing all possible values of (coef0, C, degree) for poly kernel
    kernel_type = 'poly'
    d={}
    d['kernel'] = kernel_type
    print '\nKernel:', kernel_type
    print ""
    r = {}
    for coef0 in coef0_range:
        d['coef0'] = coef0
        for C in C_range:
            d['C'] = C
            for degree in degree_range:
                d['degree'] = degree
                acc = []
                # Performing K-fold cross validation test
                for train, test in k_fold:
                    model = OneVsRestClassifier(svm.SVC(**d))
                    model.fit(train_X[train], train_Y[train])
                    pred = model.predict(train_X[test])
                    acc.append(get_accuracy(pred, train_Y[test]))
                r[(coef0,degree,C)] = list(acc)
                print 'coef0',coef0,'degree =',degree,'C =',C,'Avg.acc =',np.mean(acc)
    res[kernel_type] = r.copy()

    # Analyzing all possible values of (C, gamma) for rbf kernel
    kernel_type = 'rbf'
    d={}
    d['kernel'] = kernel_type
    print '\nKernel:', kernel_type
    print ""
    r = {}
    for C in C_range:
        d['C'] = C
        for gamma in gamma_range:
            d['gamma'] = gamma
            acc = []
            # Performing K-fold cross validation test
            for train, test in k_fold:
                model = OneVsRestClassifier(svm.SVC(**d))
                model.fit(train_X[train], train_Y[train])
                pred = model.predict(train_X[test])
                acc.append(get_accuracy(pred, train_Y[test]))
            r[(gamma,C)] = list(acc)
            print 'gamma =',gamma,'C =',C,'Avg.acc =',np.mean(acc)
    res[kernel_type] = r.copy()

    # Analyzing all possible values of (C, coef0) for sigmoid kernel
    kernel_type = 'sigmoid'
    d={}
    d['kernel'] = kernel_type
    print '\nKernel:', kernel_type
    print ""
    r = {}
    for C in C_range:
        d['C'] = C
        for coef0 in coef0_range:
            d['coef0'] = coef0
            acc = []
            # Performing K-fold cross validation test
            for train, test in k_fold:
                model = OneVsRestClassifier(svm.SVC(**d))
                model.fit(train_X[train], train_Y[train])
                pred = model.predict(train_X[test])
                acc.append(get_accuracy(pred, train_Y[test]))
            r[(coef0,C)] = list(acc)
            print 'coef0 =',coef0,'C =',C,'Avg.acc =',np.mean(acc)
    res[kernel_type] = r.copy()

    # Returning the results
    return res

def get_func_params(kernel_type, config):
    """
    A utility function to return function parameters for initiating
    the SVM model, given the kernel type and configuration.
    """
    d = {}
    d['kernel'] = kernel_type
    if kernel_type == 'linear':
        d['C'] = config[0]
    elif kernel_type == 'poly':
        d['coef0'] = config[0]
        d['degree'] = config[1]
        d['C'] = config[2]
    elif kernel_type == 'rbf':
        d['gamma'] = config[0]
        d['C'] = config[1]
    else:
        d['coef0'] = config[0]
        d['C'] = config[1]
    return d

def extract_best_fit(res):
    """
    A utility function to extract the best-fit parameters given the results
    of analyze method.
    It returns the function parameters for model initialization, for each kernel-type.
    """
    ret = {}
    for k,v in res.iteritems():
        best_fit = max([ (kk, np.mean(v[kk])) for kk in v.keys() ], key=(lambda x: x[1]))
        ret[k] = get_func_params(k, best_fit[0])
    return ret

def report_best_fit(best_fit, output_path):
    """
    A utility function for reporting performance of the best-fit model.
    It also saves the model in the output dir.
    """
    print '\nBest model predictions\n'
    print 'Kernel'.rjust(10), 'Accuracy'.rjust(10), 'Params'

    for k,v in best_fit.iteritems():
        model = OneVsRestClassifier(svm.SVC(**v))
        # Fitting the model
        model.fit(train_X, train_Y)

        # Saving the model
        joblib.dump(model, output_path + os.sep + 'model' + str(kernel_list.index(k) + 1) + '.pkl')

        # Predicting labels
        test_Y_pred = model.predict(test_X)
        # Reporting accuracy
        print str(k).rjust(10), get_accuracy(test_Y_pred, test_Y), '%', v

# Train data, test data path extracted from command-line
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
output_path = sys.argv[3]

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']

# Reading train, test data
train_data = read_data(train_data_path)
test_data = read_data(test_data_path)

# Shuffling train data
np.random.shuffle(train_data)

# Defining a min-max-transformer => Improves performance
transformer = MinMaxScaler()

# train_X = train_data[:, :-1]
# Fitting the transformer
train_X = transformer.fit_transform(train_data[:, :-1])
train_Y = np.ascontiguousarray(train_data[:, -1])

# test_X = test_data[:, :-1]
# Transforming the test-data
test_X = transformer.transform(test_data[:, :-1])
test_Y = np.ascontiguousarray(test_data[:, -1])

write_data(zip(train_X, train_Y), train_data_path.replace('.csv', '_norm.csv'))
write_data(zip(test_X, test_Y), test_data_path.replace('.csv', '_norm.csv'))

# Initiating K-fold cross validation
k_fold = KFold(len(train_X), n_folds = 5)

# Analyzing SVM performance for estimating optimal kernel parameters
res = analyze()

# Extracting the best-fit kernel parameters after analysis
best_fit = extract_best_fit(res)

# Reporting the performance of the best-fit model
report_best_fit(best_fit, output_path)
