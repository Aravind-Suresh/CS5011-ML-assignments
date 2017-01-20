import sys,os
import numpy as np
from util2 import *
from util1 import *
from sklearn import linear_model

"""
Program to demonstrate linear regression
Sample run:
    $ python linear_regression.py ../dataset/CandC 5
"""

def linear_regressor_learn(train_set):
    """
    Function to learn the linear regressor and return the model learnt
    """
    # Extracting X
    X = train_set[:,:-1]

    # Extracting labels
    Y = train_set[:,-1]

    # Training a linear regressor
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    return regr

def report_accuracy(test_set, model):
    """
    A utility function to report accuracy
    """
    # Extracting X
    X = test_set[:,:-1]

    # Extracting labels
    Y = test_set[:,-1]
    residual_err = np.sum((model.predict(X) - Y) ** 2)
    return residual_err

def main():
    fin_name_prefix = sys.argv[1]
    num = eval(sys.argv[2])

    regr_prop = []

    for i in range(1, num+1):
        train_set = np.array(read_data(fin_name_prefix + '-train' + str(i) + '.csv'), dtype = np.float32)
        test_set = np.array(read_data(fin_name_prefix + '-test' + str(i) + '.csv'), dtype = np.float32)
        model = linear_regressor_learn(train_set)
        residual_err = report_accuracy(test_set, model)
        regr_prop.append((model.coef_, residual_err))

    # Finding average residual error
    avg_residual_err = np.mean(map(lambda x: x[1], regr_prop))
    best_fit_idx = np.argmin(map(lambda x: x[1], regr_prop))

    # Finding the best fit coefficients
    best_fit_coeff = regr_prop[best_fit_idx][0]
    print 'Residual err:', avg_residual_err
    print 'Best fit coeff:', best_fit_coeff

    write_coeff_to_csv('../report/coeffs/linear_regression_best-fit_coeff.csv', best_fit_coeff)

if __name__ == '__main__':
    main()
