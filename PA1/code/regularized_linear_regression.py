import sys,os
import numpy as np
from util1 import *
from util2 import *
from sklearn import linear_model

"""
Program to demonstrate a linear regressor with regularisation

Sample run:
    $ python regularized_linear_regression.py ../dataset/CandC 5
"""

def linear_regressor_learn(train_set, reg_param):
    """
    Function to learn the linear regressor with regularisation and return the model learnt
    """
    # Extracting X
    X = train_set[:,:-1]

    # Extracting labels
    Y = train_set[:,-1]

    # Training a linear regressor
    regr = linear_model.Ridge(alpha = reg_param)
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

def data_filter(data, idx = None):
    """
    A utility function for filtering data, given indices
    """
    if idx is None:
        return data
    else:
        A = data[:, :-1]
        B = data[:, -1]
        return np.hstack((A[:, idx], B.reshape((-1, 1))))

def main_(fin_name_prefix, num, reg_param_vals, filter_idx = None):
    regr_prop = []

    for v in reg_param_vals:
        regr_prop_u = []

        for i in range(1, num+1):
            train_set = data_filter(np.array(read_data(fin_name_prefix + '-train' + str(i) + '.csv'), dtype = np.float32), filter_idx)
            test_set = data_filter(np.array(read_data(fin_name_prefix + '-test' + str(i) + '.csv'), dtype = np.float32), filter_idx)
            model = linear_regressor_learn(train_set, v)
            residual_err = report_accuracy(test_set, model)
            regr_prop_u.append((model.coef_, residual_err))

        # Measuring the residual error
        avg_residual_err = np.mean(map(lambda x: x[1], regr_prop_u))
        best_fit_idx = np.argmin(map(lambda x: x[1], regr_prop_u))
        best_fit_coeff = regr_prop_u[best_fit_idx][0]

        regr_prop.append((best_fit_coeff, avg_residual_err))

    for i in range(len(reg_param_vals)):
        print reg_param_vals[i], regr_prop[i][1]#, regr_prop[i][0]


    # Obtaining the best-fit coefficients
    best_fit_idx = np.argmin(map(lambda x: x[1], regr_prop))
    best_fit_coeff = regr_prop[best_fit_idx][0]
    avg_residual_err = regr_prop[best_fit_idx][1]

    # Obtaining the regularisation parameter for the best fit
    best_fit_lambda = reg_param_vals[best_fit_idx]

    return best_fit_lambda, avg_residual_err, best_fit_coeff

def main():
    # Getting inputs
    fin_name_prefix = sys.argv[1]
    num = eval(sys.argv[2])

    reg_param_vals = np.logspace(1, 9, 18)*1e-5

    best_fit_lambda, avg_residual_err, best_fit_coeff = main_(fin_name_prefix, num, reg_param_vals)

    print 'Best fit for lambda =', best_fit_lambda
    print 'Residual err:', avg_residual_err
    write_coeff_to_csv('../report/coeffs/regularized_linear_regression_best-fit_coeff.csv',  best_fit_coeff)

    # Reducing features based on a threshold on weights
    reduced_idx = (np.abs(best_fit_coeff) > 0.05)

    best_fit_lambda, avg_residual_err, best_fit_coeff = main_(fin_name_prefix, num, reg_param_vals, reduced_idx)

    print 'Reduced features:', np.uint8(reduced_idx)
    print 'Best fit for lambda =', best_fit_lambda
    print 'Residual err:', avg_residual_err

    write_coeff_to_csv('../report/coeffs/regularized_linear_regression_best-fit_coeff_reduced_features.csv',  best_fit_coeff)

if __name__ == '__main__':
    main()
