from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys
import os.path
import platform


SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
FileDest_FOLDER='/home/Speaker/Speech/Classification_Utternace'

DST_File1 =FileDest_FOLDER+'/classification_sample_train'
DST_File2 =FileDest_FOLDER+'/classification_sample_test'

Train_File=FileDest_FOLDER+'/classification_utterance_train'
Test_File=FileDest_FOLDER+'/classification_utterance_test'

Train_File_map=FileDest_FOLDER+'/classification_utterance_train_map'
Test_File_map=FileDest_FOLDER+'/classification_utterance_test_map'

MAP_File=FileDest_FOLDER+'/classification_utterance_map'


print(1.0/10.0)
print( platform.python_version())

if not os.path.exists(FileDest_FOLDER):
    os.makedirs(FileDest_FOLDER)


c=0
# folder_dict = dict.fromkeys(U)
folder_dict = {}
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in dirs:


            file_name=root+'/'+file
            ID=file_name.split('/')[-1]
            ID=ID.split('_')[1]
            ID=ID.split('~')[0]

            if ID in dict.keys(folder_dict):
                folder_dict[ID].append(file)
            else:
                folder_dict[ID]=[file]

            c+=1
            print(c)


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






p=0.1
c=0

cmin=60
Train=[]
Test=[]
for d in folder_dict:
    k=folder_dict[d]

    file_count=0

    while file_count<cmin:
        test = random.choice(folder_dict[d])
        path, dirs, files = os.walk(SRC_FOLDER+'/'+test).next()
        file_count = len(files)

    train=k[:]
    train.remove(test)
    Test.append(test)
    for t in train:
        Train.append(t)
    c+=1
    print(c)

Train=sorted(Train)
Test=sorted(Test)

file1 = open(Train_File,'w')
file2 = open(Test_File,'w')

for d in Train:
    file1.write(str(d) + '\n')


for d in Test:
    file2.write(str(d) + '\n')

file1.close()
file2.close()



file1 = open(Train_File_map,'w')
file2 = open(Test_File_map,'w')

for d in Train:


    # ID = d.split('/')[-1]
    ID = d.split('_')[1]
    ID = ID.split('~')[0]
    file1.write(str(d) + ' '+str(classnamestoids[ID])+'\n')



for d in Test:
    ID = d.split('/')[-1]
    ID = ID.split('_')[1]
    ID = ID.split('~')[0]
    file2.write(str(d) + ' '+str(classnamestoids[ID])+'\n')

file1.close()
file2.close()



c=0
file1 = open(DST_File1,'w')

for t in Train:
    for root, dirs, files in os.walk(os.path.join(SRC_FOLDER,t)):
        for file in files:
            file_name=root+'/'+file
            file_name=file_name.split('/')[-2]+'/'+file_name.split('/')[-1]

            ID=file_name.split('/')[-1]
            ID=ID.split('_')[1]
            ID=ID.split('~')[0]


            file1.write(file_name+' '+str(classnamestoids[ID])+ '\n')

            c=c+1
            if c%100==1:
                print(c)
file1.close()

c=0
file2 = open(DST_File2,'w')

for t in Test:
    for root, dirs, files in os.walk(os.path.join(SRC_FOLDER, t)):
        for file in files:
            file_name = root + '/' + file
            file_name = file_name.split('/')[-2] + '/' + file_name.split('/')[-1]

            ID = file_name.split('/')[-1]
            ID = ID.split('_')[1]
            ID = ID.split('~')[0]

            file2.write(file_name + ' ' + str(classnamestoids[ID]) + '\n')

            c = c + 1
            if c % 100 == 1:
                print(c)
file2.close()