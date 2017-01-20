import sys,os
import numpy as np
import sklearn
import csv
from util import *
from weka.classifiers import Classifier
from weka.core.converters import Loader
import weka.core.jvm as jvm

if not jvm.started:
    jvm.start()

"""
http://www.cs.waikato.ac.nz/ml/weka/mooc/dataminingwithweka/transcripts/Transcript3-5.txt

My understanding is that the minimum instances per leaf guarantees that at each split, at least 2 of the branches (but not necessarily more than 2) will have the minimum number of instances.

This is a sensible design. Consider an extreme case where each node has up to 10 different branches. It would require the parent node to have at least 10 times the minimum number of instances per leaf to branch! Given that the data is likely to be highly unevenly distributed among branches, we're probably more looking in the order of 50 times.

Another way to look at it is that branches is a way to separate out data. Separating one instance from 100 instances doesn't give you much information, so you set a minimum amount of separation. However, if you have a node with four branches, and two of them end up with 0 instances, the other two with 50 each, the branching still produced information.

So in one sentence, the minimum number of instances per leaf is better thought of as "the minimum amount of data separation per branching", in the case of multiway trees.
"""

# M - minNumObj
# reducedErrorPruning = False
# M<=24
# === Confusion Matrix ===
#    a   b   <-- classified as
#    464   0 |   a = e
#      0 660 |   b = p

# M>=25
# === Confusion Matrix ===
#    a   b   <-- classified as
#    464   0 |   a = e
#      7 653 |   b = p

# reducedErrorPruning = True
# M<=15
# === Confusion Matrix ===
#    a   b   <-- classified as
#    464   0 |   a = e
#      0 660 |   b = p
# M>=16
# === Confusion Matrix ===
#    a   b   <-- classified as
#    464   0 |   a = e
#      7 653 |   b = p


loader = Loader(classname="weka.core.converters.ArffLoader")

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]

test_data = loader.load_file(test_data_path)

classifier = Classifier(name="weka.classifiers.trees.J48")
classifier.train(train_data_path)
pred = classifier.predict(test_data_path)
p = []
while pred.gi_frame is not None:
    try:
        a = pred.next()
        p.append(a.predicted)
    except:
        break
