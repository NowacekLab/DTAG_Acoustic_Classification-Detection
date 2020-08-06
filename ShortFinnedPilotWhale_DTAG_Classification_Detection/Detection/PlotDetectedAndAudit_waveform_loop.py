#inputs are the whole DTAG recording (.wav file), a textfile of detected events (get before), and a text file of the audit events
#Plots whole .wav file and highlights areas where sounds were detected, sounds from the audit file, and the overlap of the two (for visualization)
#Calculates the percision, recall and accuracy 

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:29:33 2019

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


def plotWaveform(filename):
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
        
#        # envelope
#        analytic_signal = hilbert(data)
#        data_envelope = np.abs(analytic_signal)
        
        # plot
        #plot .wav file
        #x = range(0, len(data), 1)
        #xScaled = np.true_divide(x, rate)
        
        #plt.subplot(4, 1, 1)
        #plt.plot(xScaled, data, color = 'black')
#        plt.plot(np.true_divide(times[:len(data):1000], framerate), data_envelope[::1000], 'k-')
        librosa.display.waveplot(data / 2**15, sr=framerate,
                                 max_points=1000,
                                 x_axis='time',
                                 offset=times[0]/framerate,
                                 color=[0,0,0])
        plt.xlim([0, times[-1]/framerate])
        # plt.draw()
        times += nsamples
    
    f.close()
    #plt.show()

#calculate overlap
# a = audit interval [start,end]
# b = detection interval [start,end]
def getOverlap(a, b):
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    overlap =  end - start
    if overlap<0:
        return -1, -1
    else: 
        if (start == a[0]):
            returnStart = a[0]
        else:
            returnStart = b[0]
        if (end == a[1]):
            returnEnd = a[1]
        else: 
            returnEnd = b[1]
        return returnStart, returnEnd 
def generatePlots(directory, audioFile, detectionFile, auditFile, toPlot=False):
    print("generating plots...")
    os.chdir(directory) 
    nameParse = detectionFile.split('.', -1)
    figureName = nameParse[0] + '_plots.png'
    
    # statistics
    total_detection = 0
    total_audit = 0
    total_overlap = 0
    
    #originial file reading and plotting
    #read .wav file
    #rate, data = wav.read(audioFile)
    
    #plot .wav file
    #plt.clf()
    #x = range(0, len(data), 1)
    #xScaled = np.true_divide(x, rate)
    
    #plt.subplot(4, 1, 1)
    #plt.plot(xScaled, data, color = 'black')
    
    #save, show, & clear plot
    #plt.savefig('doorKnock.png')
    #plt.show()
    #plt.cla()
    #plt.clf()
    plt.subplots_adjust(hspace=.4)
    
    #plot detections
    print('plotting detections...')
    plt.subplot(3, 1, 1)
    #plt.plot(xScaled, data, color = 'black')
    if toPlot:
        plotWaveform(audioFile)
    detectionFile = open(detectionFile, 'r')    
    print("opened detection file")
    detectionStarts = []
    detectionEnds = []
    print('plotting detection regions')
    for line in detectionFile:
        #split line from text file into entries (detection number, start time(s), end time (s))
        entries = line.split(" ", -1)
        detectionNumber = float(entries[0])
        startTime = float(entries[1])
        endTime = float(entries[2])
        total_detection += (endTime - startTime)
        
        #append to list of start and end times
        detectionStarts.append(startTime)
        detectionEnds.append(endTime)
        
        #print('Event Detected:', detectionNumber)
        #print('Start time is:', startTime)
        #print('End time is:', endTime)
        
        plt.axvspan(startTime, endTime, alpha=0.5, color='green')
        #save, show, & clear plot
    #    plt.savefig('doorKnock_detections.png')
    #    plt.show()
    #    plt.cla()
    #    plt.clf()    
    plt.xlabel('')
    
    #plot audit
    print('plotting audit data...')
    plt.subplot(3, 1, 2)
    #plt.plot(xScaled, data, color = 'black')
    if toPlot:
        plotWaveform(audioFile)
    auditFile = open(auditFile, 'r')    
    print("opened audit file")
    auditStarts = []
    auditEnds = []
    print('plotting audit regions')
    for line in auditFile:
            
            #split line from text file into entries (start time(s), duration(s), call type)
            entries = line.split("\t", -1)
            startTime = float(entries[0])
            duration = float(entries[1])
            callType = entries[2]
            total_audit += duration
    
            #append to list of start and end times
            auditStarts.append(startTime)
            auditEnds.append(startTime+duration)
            
        
            #print('Start time is:', startTime)
            #print('duration is:', duration)
            #print('Call type is:', callType)
            
            plt.axvspan(startTime, startTime+duration, alpha=0.5, color='blue')
            #save, show, & clear plot
    #        plt.savefig('doorKnock_audit.png')
    #        plt.show()
    #        plt.cla()
    #        plt.clf()
    plt.xlabel('')
    
    #calculate and plot overlap
    print('plotting overlaps')
    plt.subplot(3, 1, 3)
    #plt.plot(xScaled, data, color = 'black')
    if toPlot:
        plotWaveform(audioFile)
    #compare every audit clip to every detection clip
    overlapStarts = []
    overlapEnds = []
    overlapDuration = []
    print('plotting overlap regions')
    for i in range(len(auditStarts)):
        for j in range(len(detectionStarts)):
            #append to list of overlap start and end times
            a = [auditStarts[int(i)], auditEnds[int(i)]]
            b = [detectionStarts[int(j)], detectionEnds[int(j)]]
            overlapStart, overlapEnd = getOverlap(a, b)
            if (overlapStart != -1 and overlapEnd != -1):
                overlapStarts.append(overlapStart)
                overlapEnds.append(overlapEnd)
                overlapDuration.append(overlapEnd-overlapStart)
                total_overlap += overlapEnd - overlapStart
                
                #print('Start of overlap is:', overlapStart)
                #print('End of overlap is:', overlapEnd)
                #print('Duration of overlap is:', overlapEnd-overlapStart)
            
                plt.axvspan(overlapStart, overlapEnd, alpha=0.5, color='magenta')
                #save, show, & clear plot
    #            plt.savefig('doorKnock_overlap.png')
    #            plt.show()
    #            plt.cla()
    #            plt.clf() 
    
    
    # statistics
    f = wave.open(audioFile)
    total_time = f.getnframes() / f.getframerate() # get total time
    f.close()
        
    precision = total_overlap / total_detection
    recall = total_overlap / total_audit
    accuracy = 1 - (total_detection + total_audit - 2*total_overlap) / total_time
    print('precision = {:.3f}%'.format(precision*100))
    print('recall    = {:.3f}%'.format(recall*100))
    print('accuracy  = {:.3f}%'.format(accuracy*100))
    statistics = np.array([total_detection, total_audit, total_overlap, precision, recall, accuracy])
    np.savetxt(nameParse[0]+'_statistics.txt', statistics)
    
    #save, show, & clear plot
    if toPlot:
        print('saving and displaying figure')
        plt.savefig(figureName, dpi=600)
        plt.show()
        plt.cla()
        plt.clf()
    
    os.chdir('..') 
    return statistics

#generatePlots('C:/Users/jpan2/Desktop/Detection', 'doorKnock.wav', 'doorKnock.txt', 'audit_doorKnock.txt')
generatePlots('./Detection_Test', 'F_gm188a01.wav', 'F_gm188a01_30.txt', 'gm10_188aaud.txt', toPlot=True)
generatePlots('./Detection_Test', 'F_gm188a01.wav', 'F_gm188a01_40.txt', 'gm10_188aaud.txt', toPlot=True)
generatePlots('./Detection_Test', 'F_gm188a01.wav', 'F_gm188a01_50.txt', 'gm10_188aaud.txt', toPlot=True)
#generatePlots('F_gm188a01.wav','F_gm188a01_e30.txt','gm10_188aaud.txt')

#statistics = []
#filenames = ['F_gm188a01_{:02d}.txt'.format(i) for i in range(30, 51, 2)]
#for filename in filenames:
#    statistics.append(
#        generatePlots('./Detection_Test', 'F_gm188a01.wav', filename, 'gm10_188aaud.txt', toPlot=False)
#    )
#
#np.savetxt('statistics.txt', statistics)
