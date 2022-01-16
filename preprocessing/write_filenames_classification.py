from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys


SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
DST_File1 ='/home/Speaker/Speech/Classification/classification_train_relativeaddress_map'
DST_File2 ='/home/Speaker/Speech/Classification/classification_test_relativeaddress_map'
MAP_File='/home/Speaker/Speech/Classification/classification_map'


c=0
id=[]
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in files:
            file_name=root+'/'+file

            ID=file_name.split('/')[-1]
            ID=ID.split('_')[1]
            ID=ID.split('~')[0]
            id.append(ID)
            c+=1
            if c%10000==0:
                print(c)


U=set(id)
print(len(U))
U=list(U)
U=sorted(U)
classnamestoids=dict(zip(U,range(len(U))))

file1 = open(MAP_File, 'w')
for d in sorted(classnamestoids):
    file1.write(str(d) + ' ' + str(classnamestoids[d]) + '\n')

file1.close()


file1 = open(DST_File1,'w')
file2 = open(DST_File2,'w')
p=0.1
c=0
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in files:
            file_name=root+'/'+file
            file_name=file_name.split('/')[-2]+'/'+file_name.split('/')[-1]

            ID=file_name.split('/')[-1]
            ID=ID.split('_')[1]
            ID=ID.split('~')[0]

            if random.random()>p:
                file1.write(file_name+' '+str(classnamestoids[ID])+ '\n')
            else:
                file2.write(file_name + ' ' + str(classnamestoids[ID])+ '\n')

            c=c+1
            if c%100==1:
                print(c)
file1.close()
file2.close()
