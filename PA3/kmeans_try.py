import numpy as np
import csv


def purity_score(clusters, classes):
    A = np.c_[(clusters,classes)]
    n_accurate = 0.
    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    return n_accurate / A.shape[0]

x=[]
y=[]

with open('dataset/dump-csv/D31.txt.csv','rb') as f:
    re=csv.reader(f);re.next()
    for r in re:
        x.append(map(float,r[:2]))
        y.append(float(r[2]))
x=np.array(x)
y=np.uint8(y)
from sklearn.cluster import KMeans

c=[2,4,8,16,24,32,48,64,96,128,192,256,384,512]

for cc in c:
    km=KMeans(n_clusters=cc)
    km.fit(x)
    print 'clusters =', cc, 'purity_score =', purity_score(km.labels_,y)
