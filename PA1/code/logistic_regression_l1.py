import os, sys
from util1 import *
import numpy as np

"""
Program to demonstrate logistic regression

Sample run:
    $ python logistic_regression_l1.py ../dataset/DS2/data_students/Test_labels ../dataset/DS2/out
"""

# Path to test labels
test_labels_path = sys.argv[1]
# Path to results generated after running the l1_logreg script: "logistic_regression_l1_run.sh"
results_path = sys.argv[2]

ex = open(test_labels_path,'r').readlines()
ex = np.array([int(i.strip()) for i in ex[2:]])
r = os.listdir(results_path)
arr = []

for rr in r:
    p = results_path + os.sep + rr
    pp = open(p, 'r').readlines()[7:]
    pp = np.array([float(o.strip()) for o in pp])
    arr.append((float(rr.split('_')[1]), pp))

# arr consists of (lambda, predicted)
arr.sort(key = (lambda x: x[0]))
accuracy, precision, recall, f_score = [], [], [], []
labels = list(set(ex))

for g in arr:
    a, p, r, f_s = [], [], [], []
    for label in labels:
        a.append(measure_accuracy(g[1], ex))
        pp = measure_precision(g[1], ex, label)
        p.append(pp)
        rr = measure_recall(g[1], ex, label)
        r.append(rr)
        f_s.append(measure_f_score(pp, rr))
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f_score.append(f_s)

# Best classification for highest f_score
best_idx = np.argmax(map(lambda x: (x[0]+x[1])*0.5, f_score))
best_lambda = arr[best_idx][0]
print 'Best-fit-lambda =', best_lambda
print 'Class:-1'
print 'A =', accuracy[best_idx][0], 'P =', precision[best_idx][0], 'R =', recall[best_idx][0], 'F =', f_score[best_idx][0]
print 'Class:1'
print 'A =', accuracy[best_idx][1], 'P =', precision[best_idx][1], 'R =', recall[best_idx][1], 'F =', f_score[best_idx][1]
