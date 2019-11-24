import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import h5py
import time
import pylab
import timeit
import datetime
ts = datetime.datetime.now().timestamp()
start = timeit.default_timer()
 
resistance = 500
port1 = 5
port2 = 6
fs=2**16 #sample frequency
NAverage = []
amp_gain = 1e3
number_of_samples_per_channel = 2**16
sample_Cnt = np.arange(number_of_samples_per_channel)
t = np.arange(number_of_samples_per_channel)/fs
NSum= 10
i = 0 
freqs = np.fft.rfftfreq(t.shape[-1])
cc = False
Ldata = []
time=[]
segment_size = 1 #The number of summations we wish to look at the averages for



file_n= 'data_N_{0}_R_{1}_port{2}_port{3}.npy'.format(NSum,resistance,port1,port2)
file_t= 'time_N_{0}_R_{1}_port{2}_port{3}.npy'.format(NSum,resistance,port1,port2)


def FFT_SpecConv(x,i):
    y = 2*np.sqrt(2*np.sqrt(np.multiply(x,np.conj(x))/float(i**2))/(fs*1.5/number_of_samples_per_channel))
    return y

def FFT_hanning_Normalised(d,N):
    z = np.fft.fft(np.multiply(np.hanning(N),d),norm = None)/N
    return z
    

for i in range(1,NSum+1):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai{0}".format(port1))
        task.ai_channels.add_ai_voltage_chan("Dev1/ai{0}".format(port2))
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan=number_of_samples_per_channel)
        data = np.array(task.read(number_of_samples_per_channel))/amp_gain
        
        
        z_1= np.conj(FFT_hanning_Normalised(data[0],number_of_samples_per_channel)) #Take the complex conjugate of the Fourier transform of signal 1
        z_2 = FFT_hanning_Normalised(data[1],number_of_samples_per_channel) #Take the fourier transform of 2
        cc_0 = np.multiply(z_1,z_2) #Product of the dot product of the two    
        Ldata.append(cc_0[0:int((number_of_samples_per_channel/2)+1)])
        time.append(ts)

#np.save(file_n,Ldata)
#np.save(file_t,time)
  
#print(Ldata[0])
#print(Ldata[1])

stop = timeit.default_timer()
print('Time: ', stop - start)

for j in range(1,int(NSum/segment_size)):
    for i in np.arange(j*segment size)):
        results = []
        t = []
        a = np.average(FFT_SpecConv(Ldata[i][110:140],1))
        results.append(a)
        t.append(time[j*segment_size])
    
         
      
pylab.loglog(t, results ,'r', label = 'FFT')
pylab.legend()
pylab.xlabel('Frequency [Hz]')
pylab.ylabel('V/sqrt(Hz)')
pylab.title("Cross correlation with {0} summations".format(NSum))
pylab.show()


pylab.loglog(np.arange(NSum), averages , 'bo')
pylab.xlabel('Summation Number')
pylab.ylabel('Average noise(V/sqrt(Hz)')
pylab.title('Average noise in 110-140 range vs number of summations')
pylab.show()
                            

