from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys
import os.path
import shutil


pairs_per_folder=10
n_test=50
cmin=75

SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
FileDest_FOLDER='/home/Speaker/Speech/verification_Utternace_Phone2Phone'

SRC='/home/Speaker/Speech/verification_Utternace'
SRC_File_train='/home/Speaker/Speech/verification_Utternace/verification_utterance_train_map'
SRC_File_test='/home/Speaker/Speech/verification_Utternace/verification_utterance_test_map'

DST_File1 =FileDest_FOLDER+'/verification_sample_train'
DST_File2 =FileDest_FOLDER+'/verification_sample_test'
MAP_File=FileDest_FOLDER+'/verification_utterance_map'
Train_File=FileDest_FOLDER+'/verification_ID_train'
Test_File=FileDest_FOLDER+'/verification_ID_test'
Train_File_map=FileDest_FOLDER+'/verification_utterance_train_map'
Test_File_map=FileDest_FOLDER+'/verification_utterance_test_map'


# mp3==>phone==>Ac
# ac3==>DVR==>In
# wav==>MIC==>Sm


Channel1='_Ac_'
Channel2='_Ac_'




with open(SRC_File_train) as f:
    Utterances_train = f.readlines()

Utterances_train = [x.strip() for x in Utterances_train]


with open(SRC_File_test) as f:
    Utterances_test = f.readlines()


Utterances_test = [x.strip() for x in Utterances_test]


if not os.path.exists(FileDest_FOLDER):
    os.makedirs(FileDest_FOLDER)

# shutil.copy2(SRC+'/verification_utterance_map', FileDest_FOLDER)
# shutil.copy2(Train_File, FileDest_FOLDER)
# shutil.copy2(Test_File, FileDest_FOLDER)


c=0
folder_dict_train = {}
# dirs=os.listdir(SRC_FOLDER)
id=[]
for file in Utterances_train:
    if Channel1 in file or Channel2 in file:
        file_name=SRC_FOLDER+'/'+file
        ID=file_name.split('/')[-1]
        ID=ID.split('_')[1]
        ID=ID.split('~')[0]
        id.append(ID)

        if ID in dict.keys(folder_dict_train):
            folder_dict_train[ID].append(file)
        else:
            folder_dict_train[ID]=[file]

        c+=1
        if c%100==0:
            print(c)


U=set(id)
print(len(U))
U=list(U)
U=sorted(U)
Train_id_0=U[:]


########################################################################################################################
c=0
folder_dict_test = {}
# dirs=os.listdir(SRC_FOLDER)
id=[]
for file in Utterances_test:
    if Channel1 in file or Channel2 in file:

        file_name=SRC_FOLDER+'/'+file
        ID=file_name.split('/')[-1]
        ID=ID.split('_')[1]
        ID=ID.split('~')[0]
        id.append(ID)

        if ID in dict.keys(folder_dict_test):
            folder_dict_test[ID].append(file)
        else:
            folder_dict_test[ID]=[file]

        c+=1
        if c%100==0:
            print(c)


U=set(id)
print(len(U))
U=list(U)
U=sorted(U)
Test_id_0=U[:]

Train_id=[]
for t in Train_id_0:
    f1=0
    f2=0
    K=folder_dict_train[t]
    for k in K:
        if Channel1 in k:
            f1=1
        if Channel2 in k:
            f2=1

    if f1*f2==0:
        del folder_dict_train[t]
    else:
        Train_id.append(t)


    c+=1
    print(c)

Train_id=sorted(Train_id)

Test_id = []
for t in Test_id_0:
    f1 = 0
    f2 = 0
    K = folder_dict_test[t]
    for k in K:
        if Channel1 in k:
            f1 = 1
        if Channel2 in k:
            f2 = 1

    if f1 * f2 == 0:
        del folder_dict_test[t]
    else:
        Test_id.append(t)

    c += 1
    print(c)

Test_id = sorted(Test_id)



U=Test_id+Train_id
U=sorted(U)
classnamestoids=dict(zip(U,range(len(U))))

file1 = open(MAP_File, 'w')
for d in sorted(classnamestoids):
    file1.write(str(d) + ' ' + str(classnamestoids[d]) + '\n')

file1.close()



file1 = open(Train_File,'w')
file2 = open(Test_File,'w')

for d in Train_id:
    file1.write(str(d) + '\n')


for d in Test_id:
    file2.write(str(d) + '\n')

file1.close()
file2.close()
#########################################################################################

file1 = open(Train_File_map,'w')
file2 = open(Test_File_map,'w')

for d in Train_id:
    for t in folder_dict_train[d]:
        file1.write(str(t) + '\n')


for d in Test_id:
    for t in folder_dict_test[d]:
        file2.write(str(t) + '\n')

file1.close()
file2.close()


#########################################################################################
file1 = open(DST_File1,'w')
c=0
for t in Train_id:
    K=folder_dict_train[t]
    for n in range(pairs_per_folder):
        for k in K:
            if Channel1 in k:

                _, _, files = os.walk(SRC_FOLDER+'/'+k).next()
                pg1=k+'/'+random.choice(files)
                P1=''
                while not (Channel2 in P1):
                    P1=random.choice(K)
                _, _, files = os.walk(SRC_FOLDER + '/' + P1).next()
                pg2 = P1 + '/' + random.choice(files)

                _, _, files = os.walk(SRC_FOLDER + '/' + k).next()
                pi1 = k + '/' + random.choice(files)

                P2=Train_id[:]
                P2.remove(t)
                P2=random.choice(P2)
                Kii = folder_dict_train[P2]
                Ki=''
                while not (Channel2 in Ki):
                    Ki=random.choice(Kii)

                _, _, files = os.walk(SRC_FOLDER + '/' + Ki).next()
                pi2 = Ki + '/' + random.choice(files)

                file1.write(str(pg1) + ' ' + str(pg2) +' '+ '0' + '\n')
                file1.write(str(pi1) + ' ' + str(pi2) + ' ' + '1' + '\n')
    c+=1
    print(c)

file1.close()


file2 = open(DST_File2,'w')
c=0

for t in Test_id:
    K=folder_dict_test[t]
    for n in range(pairs_per_folder):
        for k in K:
            if Channel1 in k:

                _, _, files = os.walk(SRC_FOLDER+'/'+k).next()
                pg1=k+'/'+random.choice(files)
                P1=''
                while not (Channel2 in P1):
                    P1=random.choice(K)
                _, _, files = os.walk(SRC_FOLDER + '/' + P1).next()
                pg2 = P1 + '/' + random.choice(files)

                _, _, files = os.walk(SRC_FOLDER + '/' + k).next()
                pi1 = k + '/' + random.choice(files)

                P2=Test_id[:]
                P2.remove(t)
                P2=random.choice(P2)
                Kii = folder_dict_test[P2]
                Ki=''
                while not (Channel2 in Ki):
                    Ki=random.choice(Kii)

                _, _, files = os.walk(SRC_FOLDER + '/' + Ki).next()
                pi2 = Ki + '/' + random.choice(files)

                file2.write(str(pg1) + ' ' + str(pg2) +' '+ '0' + '\n')
                file2.write(str(pi1) + ' ' + str(pi2) + ' ' + '1' + '\n')
    c+=1
    print(c)

file2.close()


##########################################################################################33
