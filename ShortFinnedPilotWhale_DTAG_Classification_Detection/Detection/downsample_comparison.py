# Plots audio waveform with different methods of downsampling 
# (For visualization purposes

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:48:14 2019

@author: jpan2
"""
#imports
import matplotlib.pyplot as plt
import numpy as np 
from scipy.io import wavfile as wav
from scipy.signal import hilbert
import librosa
import librosa.display

import wave
import struct
import time
import os


def plotWaveform(filename, method='librosa'):
    # open file
    f = wave.open(filename)
    print("read wav file")
#    print(f.getparams())
    framerate = f.getframerate()
    nchannels  = f.getnchannels()

    
    # read from wav file
    nsamples = 5000000
    times = np.arange(nsamples)
    
    while f.tell() < f.getnframes():
        print('processing frame {:d} of {:d}'.format(f.tell(), f.getnframes()))
        bytedata = f.readframes(nsamples)
        #data = np.fromstring(bytedata, dtype='<i4') # 2 byte, little endian signed int
        
        data = np.fromstring(bytedata, dtype='<i2') # 2 byte, little endian signed int
        data = data [::f.getnchannels()]
        
        
        # plot
        #plot .wav file
        #x = range(0, len(data), 1)
        #xScaled = np.true_divide(x, rate)
        
        if method == 'none':
            plt.plot(times[:len(data)]/framerate, data, 'k-')
        elif method == 'naive':
            plt.plot(np.true_divide(times[:len(data):1000], framerate), data[::1000], 'k-')
        elif method == 'hilbert':
            # envelope
            analytic_signal = hilbert(data)
            data_envelope = np.abs(analytic_signal)
            plt.fill_between(np.true_divide(times[:len(data):1000], framerate),
                             -data_envelope[::1000],
                             y2=data_envelope[::1000],
                             color=[0,0,0])
        elif method == 'librosa':
            librosa.display.waveplot(data / 1.0, sr=framerate,
                                     max_points=1000,
                                     x_axis='time',
                                     offset=times[0]/framerate,
                                     color=[0,0,0])
        plt.xlim([0, times[len(data)]/framerate])
        # plt.draw()
        times += nsamples
    
    f.close()
    
def main():
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=1)
    plt.subplot(4,1,1)
    plt.title('No Downsampling')
    plotWaveform('doorKnock.wav', method='none')
    plt.subplot(4,1,2)
    plt.title('Naive Downsampling')
    plotWaveform('doorKnock.wav', method='naive')
    plt.subplot(4,1,3)
    plt.title('Hilbert Envelope Downsampling')
    plotWaveform('doorKnock.wav', method='hilbert')
    plt.subplot(4,1,4)
    plt.title('Librosa Downsampling')
    plotWaveform('doorKnock.wav', method='librosa')
    plt.xlabel('time (s)')
    plt.savefig('doorKnock_comparison.png', dpi=900)
    plt.show()

if __name__ == '__main__':
    main()