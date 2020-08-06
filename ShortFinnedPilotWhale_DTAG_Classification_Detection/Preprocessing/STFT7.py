#Code used to compute Short Time Fourier Transform (STFT) on .wav files 
#(extracted buzz and minibuzz events) in filepath "path" 
#STFT and labels are input for training the network 																					
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:23:28 2019

@author: jpan2
"""
import os
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def stft(path):
    os.chdir(path) # change to the file directory
    
    for filename in glob.glob(os.path.join(path, '*.wav')):
        base = os.path.basename(filename)
        nameParse = base.split(".",-1)
        ID = nameParse[0]
        #print ID
        
        data, sr = librosa.load(filename,sr=192000)
        #print sr
        dur = librosa.get_duration(y=data,sr=192000)
        #print dur
        
        D = np.abs(librosa.core.stft(data, n_fft=2048, hop_length=512, win_length=1024, window=signal.hann(1024), center=False, pad_mode='reflect'))
        
        #save STFT data
        filename = 'STFT_'+ID+'.npy'
        np.save(filename, D, allow_pickle=True, fix_imports=True)
        
        #make as save plot
        plt.clf()
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time', sr=192000)
        plt.title('Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        figName = 'Spec_'+ID+'.png'
        plt.savefig(figName)
        plt.cla()
        plt.clf()

#2008 (5)
#stft('D:\Pilotwhales2008\data\gm08\gm08_143a\Use') 
stft('D:\Pilotwhales2008\data\gm08\gm08_143b\Use')
stft('D:\Pilotwhales2008\data\gm08\gm08_147a\Use')
stft('D:\Pilotwhales2008\data\gm08\gm08_151a\Use')
stft('D:\Pilotwhales2008\data\gm08\gm08_151b\Use')

#2011 (9)
stft('D:\Pilotwhales2011\data\gm11\gm11_147a\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_148a\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_149b\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_149c\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_150b\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_155a\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_156a\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_158b\use')
stft('D:\Pilotwhales2011\data\gm11\gm11_165a\use')




