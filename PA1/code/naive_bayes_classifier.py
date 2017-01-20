import sys,os
import numpy as np

from util1 import *

from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

"""
Program to demonstrate a Naive Bayes Classifier to classify email as {spam, ham}

Sample run:
    $ python naive_bayes_classifier
"""

kf_idx = cross_validation.KFold(10, n_folds=5)

def plot_pr_curve(y_true, y_pred, head):
    """
    A utility function to plot precision vs recall curves
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall '+head+' : AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.show()

def get_words(file_path):
    """
    A utility function to obtain list of words from a text file
    """
    fin = open(file_path, 'r')
    words = fin.read().split()
    return words

def create_unique_word_list(file_path_prefix, num):
    """
    A utility function to obtain a set of unique words.
    This forms our vocabulary and feature vectors are generated from this.
    """
    unique_word_list = []
    for i in range(1, num + 1):
        path = file_path_prefix + str(i)
        for filename in os.listdir(path):
            unique_word_list.extend(get_words(path + os.sep + filename))
            unique_word_list = list(set(unique_word_list))
    unique_word_list.remove('Subject:')
    return unique_word_list

unique_word_list = create_unique_word_list('../dataset/Q10/part', 10)
print 'Num', len(unique_word_list)

def extract_data_point(file_path):
    """
    Extracts data point by obtaining frequency distribution over unique_word_list and labelling it as spam or not spam
    Label for spam is: 1 else 0
    """
    word_list = get_words(file_path)
    return ( [ word_list.count(x) for x in unique_word_list ], 1 if ('spmsg' in file_path) else 0 )

GEN = 1

def read_data(folder_prefix):
    """
    A utility function to read data, given the file path
    """
    if GEN:
        X = np.genfromtxt(folder_prefix + os.sep + 'data.csv', delimiter = ',', dtype = int)
        # print X.shape
        Y = np.genfromtxt(folder_prefix + os.sep + 'labels.csv', delimiter = ',', dtype = int)
    else:
        X, Y = [], []
        for filename in os.listdir(folder_prefix):
            if not filename.find(".txt") == -1:
                x, y = extract_data_point(folder_prefix + os.sep + filename)
                X.append(x)
                Y.append(y)
        X = np.matrix(X)
        Y = np.array(Y).astype(int)
        np.savetxt(folder_prefix + os.sep + 'data.csv', X, delimiter = ',', fmt = '%i')
        np.savetxt(folder_prefix + os.sep + 'labels.csv', Y, delimiter = ',', fmt = '%i')

    return X, Y

def multinomial_learn(X, Y):
    """
    Function to learn a multinomial distribution based maximum likelihood estimator
    """
    X = np.array(X)
    ele_label_0 = X[Y == 0, :]
    ele_label_1 = X[Y == 1, :]
    len_Y = len(Y)
    label_0_red_col = np.sum(ele_label_0, axis = 0)
    label_1_red_col = np.sum(ele_label_1, axis = 0)
    # print 68, label_0_red_col.shape, np.sum(label_0_red_col), len_Y

    p_label_0 = (label_0_red_col + 1)/(1.0*np.sum(label_0_red_col) + len_Y)
    p_label_1 = (label_1_red_col + 1)/(1.0*np.sum(label_1_red_col) + len_Y)

    p_1 = 1.0*np.sum(np.array(Y))/len_Y
    p_0 = 1 - p_1

    return np.log(p_0), np.log(p_1), np.log(p_label_0), np.log(p_label_1)

def multinomial_predict(X, prior_0, prior_1, p_label_0, p_label_1):
    """
    Function to predict given a maximum likelihood estimator ( multinomial )
    """
    a = prior_0
    b = prior_1
    # print X.shape, p_label_1.shape, prior_1
    p_0 = np.exp(prior_0 + np.dot(X, p_label_0))
    p_1 = np.exp(prior_1 + np.dot(X, p_label_1))
    p_spam = [ p_1[i]/(p_0[i] + p_1[i]) if(p_0[i] + p_1[i]) > 0 else 0 for i in range(len(p_0))]

    return np.uint8(p_1 > p_0), np.array(p_spam)

def bernoulli_learn(X, Y):
    """
    Function to learn a bernoulli distribution based maximum likelihood estimator
    """
    ele_label_0 = X[Y == 0, :]
    ele_label_1 = X[Y == 1, :]
    len_Y = len(Y)

    label_0_red_col = np.sum(ele_label_0, axis = 0)
    label_1_red_col = np.sum(ele_label_1, axis = 0)

    p_label_0 = (label_0_red_col + 1)/(2.0 + ele_label_0.shape[0])
    p_label_1 = (label_1_red_col + 1)/(2.0 + ele_label_1.shape[0])

    p_1 = 1.0*np.sum(Y)/len_Y
    p_0 = 1 - p_1

    return np.log(p_0), np.log(p_1), np.log(p_label_0), np.log(p_label_1)

def bernoulli_predict(X, prior_0, prior_1, p_label_0, p_label_1):
    """
    Function to predict given a maximum likelihood estimator ( bernoulli )
    """
    l_p_label_0 = np.log(p_label_0)
    l_p_label_0_ = np.log(np.ones_like(p_label_0) - p_label_0)

    s_0 = prior_0 + np.dot(X, l_p_label_0 + l_p_label_0_)

    l_p_label_1 = np.log(p_label_1)
    l_p_label_1_ = np.log(np.ones_like(p_label_1) - p_label_1)

    s_1 = prior_1 + np.dot(X, l_p_label_1 + l_p_label_1_)

    p_0 = np.exp(s_0)
    p_1 = np.exp(s_1)
    p_spam = [ p_1[i]/(p_0[i] + p_1[i]) if(p_0[i] + p_1[i]) > 0 else 0 for i in range(len(p_0))]

    return np.uint8(p_1 > p_0), np.array(p_spam)

def dirichlet_learn(X, Y, alpha):
    """
    Function to learn a dirichlet distribution based maximum likelihood estimator
    """
    alpha = 1.0*alpha
    alpha_sum = np.sum(alpha)

    ele_label_0 = X[Y == 0, :]
    ele_label_1 = X[Y == 1, :]
    len_Y = len(Y)

    label_0_red_col = np.sum(ele_label_0, axis = 0)
    label_1_red_col = np.sum(ele_label_1, axis = 0)

    p_label_0 = (label_0_red_col + alpha)/(1.0*np.sum(label_0_red_col) + alpha_sum)
    p_label_1 = (label_1_red_col + alpha)/(1.0*np.sum(label_1_red_col) + alpha_sum)

    p_1 = 1.0*np.sum(Y)/len_Y
    p_0 = 1 - p_1

    return np.log(p_0), np.log(p_1), np.log(p_label_0), np.log(p_label_1)

def beta_learn(X, Y, alpha, beta):
    """
    Function to learn a beta distribution based maximum likelihood estimator
    """
    ele_label_0 = X[Y == 0, :]
    ele_label_1 = X[Y == 1, :]
    len_Y = len(Y)

    label_0_red_col = np.sum(ele_label_0, axis = 0)
    label_1_red_col = np.sum(ele_label_1, axis = 0)

    # print alpha, beta, label_0_red_col.shape, ele_label_0.shape
    p_label_0 = (label_0_red_col + alpha)/(1.0*ele_label_0.shape[0] + alpha + beta)
    p_label_1 = (label_1_red_col + alpha)/(1.0*ele_label_1.shape[0] + alpha + beta)

    p_1 = 1.0*np.sum(Y)/len_Y
    p_0 = 1 - p_1

    return np.log(p_0), np.log(p_1), np.log(p_label_0), np.log(p_label_1)

def get_elements(idx, path = '../dataset/Q10/part'):
    """
    Function to return selective elements of data given indices
    """
    train_X = np.empty((0, len(unique_word_list)))
    # print train_X.shape
    train_Y = np.empty(0)
    for i in (idx+1):
        _path = path + str(i)
        X, Y = read_data(_path)
        # print X.shape
        train_X = np.vstack((train_X, X))
        train_Y = np.append(train_Y, Y)

    return train_X, train_Y

def compute_optimal_alpha(alpha_0):
    """
    Function to compute the optimal value of alpha for dirichlet learning
    """
    auc = []
    alpha_vals = np.array(range(10))

    kf_idx = cross_validation.KFold(10, n_folds = 5)

    for aa in alpha_vals:
        a = aa + alpha_0
        Y_true_curve = np.zeros(0)
        Y_pred_curve = np.zeros(0)

        for tr_idx, te_idx in kf_idx:
            train_X, train_Y = get_elements(tr_idx)
            test_X, test_Y = get_elements(te_idx)

            prior_0, prior_1, p_label_0, p_label_1 = dirichlet_learn(train_X, train_Y, a)
            Y_pred, p_Y_pred = multinomial_predict(test_X, prior_0, prior_1, p_label_0, p_label_1)
            Y_true_curve = np.append(Y_true_curve, test_Y)
            Y_pred_curve = np.append(Y_pred_curve, p_Y_pred)

        auc = average_precision_score(Y_true_curve, Y_pred_curve)
        return alpha_0 + alpha_vals[np.array(auc).argmax()]

def compute_optimal_alpha_beta(X):
    """
    Function to compute the optimal value of { alpha, beta } for beta learning
    """
    auc = []
    alpha_vals = np.array(range(1, 4))
    beta_vals = alpha_vals.copy()

    for a in alpha_vals:
        auc_ = []
        for b in beta_vals:
            Y_true_curve = np.zeros(0)
            Y_pred_curve = np.zeros(0)

            for tr_idx, te_idx in kf_idx:
                train_X, train_Y = get_elements(tr_idx)
                test_X, test_Y = get_elements(te_idx)

                prior_0, prior_1, p_label_0, p_label_1 = beta_learn(train_X, train_Y, a, b)
                Y_pred, p_Y_pred = bernoulli_predict(test_X, prior_0, prior_1, p_label_0, p_label_1)
                Y_true_curve = np.append(Y_true_curve, test_Y)
                Y_pred_curve = np.append(Y_pred_curve, p_Y_pred)

            auc_.append(average_precision_score(Y_true_curve, Y_pred_curve))
        auc.append(auc_)
    auc = np.matrix(auc)
    a_idx = np.where(auc == auc.max())[0][0, 0]
    b_idx = np.where(auc == auc.max())[1][0, 0]

    return alpha_vals[a_idx], beta_vals[b_idx]

def compute_pr_curve(method, alpha = None, beta = None):
    """
    Function to compute the Precision-Recall curves and plot it
    """
    print method
    accuracy, precision, recall, f_score = [], [], [], []
    Y_true_curve = np.empty(0)
    Y_pred_curve = np.empty(0)

    for tr_idx, te_idx in kf_idx:
        train_X, train_Y = get_elements(tr_idx)
        test_X, test_Y = get_elements(te_idx)

        if method == 'multinomial':
            prior_0, prior_1, p_label_0, p_label_1 = multinomial_learn(train_X, train_Y)
            Y_pred, p_Y_pred = multinomial_predict(test_X, prior_0, prior_1, p_label_0, p_label_1)

        elif method == 'bernoulli':
            train_X = (train_X.astype(bool)).astype(int)
            test_X = (test_X.astype(bool)).astype(int)
            prior_0, prior_1, p_label_0, p_label_1 = bernoulli_learn(train_X, train_Y)
            Y_pred, p_Y_pred = bernoulli_predict(test_X, prior_0, prior_1, p_label_0, p_label_1)

        elif method == 'dirichlet':
            prior_0, prior_1, p_label_0, p_label_1 = dirichlet_learn(train_X, train_Y, alpha)
            Y_pred, p_Y_pred = multinomial_predict(test_X, prior_0, prior_1, p_label_0, p_label_1)

        else:
            train_X = (train_X.astype(bool)).astype(int)
            test_X = (test_X.astype(bool)).astype(int)
            prior_0, prior_1, p_label_0, p_label_1 = beta_learn(train_X, train_Y, alpha, beta)
            Y_pred, p_Y_pred = bernoulli_predict(test_X, prior_0, prior_1, p_label_0, p_label_1)

        Y_true_curve = np.append(Y_true_curve, test_Y)
        Y_pred_curve = np.append(Y_pred_curve, p_Y_pred)
        A = measure_accuracy(Y_pred, test_Y)
        P = measure_precision(Y_pred, test_Y)
        R = measure_recall(Y_pred, test_Y)
        F = measure_f_score(P, R)

        accuracy.append(A)
        precision.append(P)
        recall.append(R)
        f_score.append(F)

    plot_pr_curve(Y_true_curve, Y_pred_curve, method)
    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f_score), average_precision_score(Y_true_curve, Y_pred_curve)

print compute_pr_curve('multinomial')

print compute_pr_curve('bernoulli')

# alpha = compute_optimal_alpha(np.ones_like(unique_word_list).astype(int))
# print "alpha =", alpha
alpha = np.ones_like(unique_word_list).astype(int) + 1
print compute_pr_curve('dirichlet', alpha)

# alpha, beta = compute_optimal_alpha_beta(unique_word_list)
# print "alpha =", alpha, "beta =", beta
alpha = beta = 2
print compute_pr_curve('beta', alpha, beta)
