import h5py
import matplotlib.pyplot as plt
import pylab 
import numpy as np

path = 'data_N_10000_R_500_port5_port6_.h5'


hf = h5py.File(path, 'r')
n1 = np.array(hf["Number_summations"][:])#dataset_name is same as hdf5 object name 
n2 = np.array(hf['Average_noise'][:])


pylab.loglog(n1,n2, 'bo')

pylab.xlabel('Summation Number')
pylab.ylabel('Average noise(V/sqrt(Hz)')
pylab.title('Average noise in 110-140 range vs number of summations')
pylab.show()



