import numpy as np

file_name='/home/Speaker/test_active.npy'


C=np.load(file_name)

outfile='testraw'

np.savetxt("test.txt", C[:,:,0])
