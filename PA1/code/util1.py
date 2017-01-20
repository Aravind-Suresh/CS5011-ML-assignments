import numpy as np

"""
Contains utility functions for syn_ds_create.py, linear_classification.py and knn_classifier.py
"""

def read_dataset(fin_train, fin_test, p):
    """
    A function to return the dataset after reading it from the files, given the path to train/test file
    """
    train_set = np.loadtxt(fin_train, delimiter=',').reshape((-1, p+1))
    test_set = np.loadtxt(fin_test, delimiter=',').reshape((-1, p+1))
    return train_set, test_set

def write_dataset(fout, data):
    """
    A utility function to write the dataset to file given the output file path and data
    """
    data.tofile(fout, sep = ',', format = '%s')

def measure_f_score(precision, recall):
    """
    A function to measure the f_score, given precision and recall
    F = 2PR/(P+R), where P = precision, R = recall and F = f_score
    """
    num = 2*precision*recall
    den = precision + recall
    # Handling division by zero case, which is caused by precision = recall = 0
    if den == 0:
        return 0
    return num/den

def measure_accuracy(pred, exact):
    """
    A function to measure the accuracy of the model given the predicted and the exact values
    Accuracy = (No. of correct predictions)/(Total predictions)
    """
    misclassified = np.count_nonzero(pred.ravel() - exact.ravel())
    total = len(pred.ravel())
    accuracy = 1.0*(total-misclassified)/total
    return accuracy

def measure_precision(pred, exact, label = 1):
    """
    A function to measure the precision of the model given the predicted and the exact values
    Precision = (No. of true positives)/(No. of total positives predicted)
    """
    _pred = pred.ravel()
    _exact = exact.ravel()
    pred_pos = np.count_nonzero(_pred == label)
    pred_pos_true = np.count_nonzero((_pred == label) & (_exact == label))
    if pred_pos == 0:
        return 0
    return 1.0*pred_pos_true/pred_pos

def measure_recall(pred, exact, label = 1):
    """
    A function to measure the recall of the model given the predicted and the exact values
    Recall = (No. of true positives)/(Total positives in the dataset)
    """
    _pred = pred.ravel()
    _exact = exact.ravel()
    exact_pos = np.count_nonzero(_exact == label)
    pred_pos_true = np.count_nonzero((_pred == label) & (_exact == label))
    return 1.0*pred_pos_true/exact_pos

def write_coeff_to_csv(file_path, coeff):
    """
    A utility function to write coefficients to a csv file for reporting purposes
    """
    fout = open(file_path, 'w')
    fout.write(','.join(map(str, coeff.tolist())))
    fout.close()
