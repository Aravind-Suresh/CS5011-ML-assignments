from matplotlib import pyplot as plt
import numpy as np
x=range(1,21)
cp=[ 0.0667  ,  0.1333  ,  0.2     ,  0.2667  ,  0.3333  ,  0.4     ,
        0.4667  ,  0.5333  ,  0.6     ,  0.6667  ,  0.7333  ,  0.8     ,
        0.86    ,  0.928333,  0.8967  ,  0.8633  ,  0.8533  ,  0.92    ,
        0.915   ,  0.898333]
f=plt.figure()
plt.title('Plot of Cluster-purity ( C ) Vs No. of clusters ( k )')
a=('No. of clusters, k', 'Cluster-purity, C')
plt.xlabel(a[0])
plt.ylabel(a[1])
plt.plot(x,cp,'ro',)
plt.plot(x,cp,'b')
plt.show()

# For k=8, cluster purity = 53.33 %
