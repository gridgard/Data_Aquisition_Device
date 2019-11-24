import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import h5py
from signal_gen import signal

resistance = 500
N=1000
port = 6
fs=2**17 #sample frequency
amp_gain = 1e3
N0= 1000.0
Pxx = False
Z=False
t=np.arange(N)/fs
i = 0
j=0
sign = signal(noise_power, freq, amp)


while (i<N0):
    i+=1
         
    z = np.conj(np.fft.fft(np.hanning(number_of_samples_per_channel)*data))
    f, P = sig.welch(data,fs, nperseg = number_of_samples_per_channel)
    if (i==0):
        Z_1 =z    
        Pxx =P
    else:
        Pxx += P
        z_1 +=z

Pxx = Pxx/N0
freqs = np.abs(np.fft.fftfreq(t.shape[-1]))

while (j<0):
    j+=1
    if f[j]==freqs[j]:
        print("Frequencies match)")
    else:
        print("Frequencies don't match")

