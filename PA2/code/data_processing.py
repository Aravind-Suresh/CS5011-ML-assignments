from PIL import Image
import glob

"""
Sample run: ( from ../dataset directory )
    $ python ../code/data_processing.py ../../PA1/dataset/DS2/data_students/
"""

def features(data):
    rows = len(data)
    cols = len(data[0])
    all_features = []
    for j in xrange(cols):
        bins = [0]*32
        for i in xrange(rows):
            bins[data[i][j]/8] += 1
        all_features.extend(bins)
    return all_features

def write_d(file_name, data):
    f = open(file_name, "w")
    # f.write(header)
    # f.write("%d %d\n"%(len(data),len(data[0])))
    for a in data:
        f.write(','.join(map(str,a))+'\n')
    f.close()

def one_hot(idx, l):
    ret=np.zeros(l)
    ret[idx]=1
    return ret
import sys
path=sys.argv[1]

splits = ["Test", "Train"]
classes = ["coast","forest","insidecity","mountain"]
labels = {}
total_classes = len(classes)
for c in range(total_classes):
    labels[classes[c]]=c#one_hot(c,total_classes)
# header = "%%MatrixMarket matrix array real general\n"
for split in splits:
    arr=[]
    for clas in classes:
        for filename in glob.glob(path+clas + "/" + split + '/*.jpg'): #assuming gif
            im=Image.open(filename)
            data = im.getdata()
            arr.append(features(data)+[labels[clas]])
    write_d('DS2_'+split.lower()+'.csv', arr)
