import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import h5py


fs=2**17 #sample frequency


number_of_samples_per_channel = 2**17

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai6")

    
    task.timing.cfg_samp_clk_timing(fs,samps_per_chan=number_of_samples_per_channel)
    data = np.array(task.read(number_of_samples_per_channel))

    print(np.shape(data))          
    
    
    sample_Cnt = np.arange(number_of_samples_per_channel)

    f, Pxx = sig.welch(data,fs, nperseg = number_of_samples_per_channel)

    with h5py.File('data.h5', 'w') as a:
        dset_f = a.create_dataset('frequencies', data = f)
        dset_p = a.create_dataset('Power_spectrum', data = Pxx)
    

    

    print(np.shape(f))

    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectrum V**2")
    plt.axis([0,fs/2,-0.01,0.01])
    plt.show()
