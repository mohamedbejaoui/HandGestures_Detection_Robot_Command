import numpy as np
import os
import sys

labels = []
files = []

for i,arg in enumerate(sys.argv):
    if(i>0):
        if(i%2 == 0):
            #we have a label
            labels.append(int(arg))
        else:
            #we have a file
            if(not os.path.isfile(arg)):
                print("{} does not exist".format(arg))
            else:
                files.append(arg)
if(len(labels) != len(files)):
    print("Data not coherente")
    sys.exit()
all_data = np.empty(shape=[0,22])
for i,f in enumerate(files):
    hand_feat = np.loadtxt(f,delimiter=',')
    label = labels[i]*np.ones((hand_feat.shape[0],1))
    hand_feat = np.append(label,hand_feat,axis = 1)
    all_data = np.append(all_data,hand_feat,axis=0)
np.savetxt('data/all_data.csv',all_data,delimiter=',')
