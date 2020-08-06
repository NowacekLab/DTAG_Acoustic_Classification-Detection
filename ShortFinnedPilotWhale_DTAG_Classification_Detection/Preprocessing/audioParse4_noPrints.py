#Code used to extarct buzz and minibuzz audio clips from DTAG audio recordings. 
#Inputs include filepath where .wav recodings, log file, and audit file are, 
#AudioClips log file that lists the name of the .wav files from an individual
# and the duration of the audio file in H:MM:SS.SSS
#and the Audit file of events for that individual

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:14:36 2019

@author: jpan2
"""
import os
from pydub import AudioSegment

def segmentAudio (path, AudioClips, AuditFile):
    
    print("starting segmentAudio method")
    print path
    print AudioClips
    print AuditFile
    
    os.chdir(path)
    
    #open AudioClipsFile (contains .wav file name and length of recording)
    audioClipsFile = open(AudioClips, 'r')
    
    print("Opened log file")
    
    #read in first audioFile entry
    line = audioClipsFile.readline() 
    entries2 = line.split("\t",-1) 
    wavFile = entries2[0]
    wavFileLength = entries2[1]
    
    #print line
    #print entries2
    #print wavFile
    #print wavFileLength
    
    #import audio file
    audioFile = AudioSegment.from_wav(wavFile)
    print("imported Audiofile")
    
    #calculate length of audioFile in milliseconds, and save as totalTimeMilli
    hours_minutes = wavFileLength.split(":",-1)
    hours = float(hours_minutes[0])
    minutes = float(hours_minutes[1])
    seconds_milliseconds = hours_minutes[2].split(".",-1)
    seconds = float(seconds_milliseconds[0])
    milliseconds = float(seconds_milliseconds[1])
    totalTimeMilli = (hours*60000*60)+(60000*minutes)+(1000*seconds)+(milliseconds)
    
    #print hours
    #print minutes
    #print seconds
    #print milliseconds
    #print totalTimeMilli
        
    #open audit file
    auditFile = open(AuditFile, 'r')    
    eventCounter = 0
    buzzCounter = 0
    minibuzzCounter = 0
    offset = 0
    
    print("opened Audit file")
    
    for line in auditFile:
        eventCounter = eventCounter+1 
        
        #print 'event number:',eventCounter
        
        #split line from text file into entries (start time(s), duration(s), call type)
        entries = line.split("\t", -1)
        startTime = float(entries[0])
        duration = float(entries[1])
        callType = entries[2]
        
        #print entries
        #print'Start time is:', startTime
        #print'duration is:', duration
        #print'Call type is:', callType
        
        #check call type- only concerned with buzzes and minibuzzes
        #if (callType == "buzz\n" or callType == "minibuzz\n"):     
            
        #convert start time and duration to milliseconds for pydub
        startTimeMilli = (startTime * 1000)
        durationMilli = duration * 1000
        endTimeMilli = startTimeMilli+durationMilli
            
        #print 'start time is:', startTimeMilli
        #print 'end time is:', endTimeMilli
        #print 'total time is:', totalTimeMilli
            
        #determine if you need to move on to nenxt audio clip
        #if endTimeMilli>running time --> get next .wav file name, append total time
        if (endTimeMilli>totalTimeMilli):
            print("going to next audio clip!")
                
            line = audioClipsFile.readline()
            entries2 = line.split("\t",-1)
            wavFile = entries2[0]
            wavFileLength = entries2[1]
                
            #print line
            #print entries2
            #print wavFile
            #print wavFileLength
            
            #import audio file
            audioFile = AudioSegment.from_wav(wavFile)
            print("imported Audiofile")
                
            #print offset
            #print totalTimeMilli
            #set new offset (subtract length of previous clips)
            offset = totalTimeMilli
            #print offset
                
            #calculate length of audioFile in milliseconds, and save as totalTimeMilli
            hours_minutes = wavFileLength.split(":",-1)
            hours = float(hours_minutes[0])
            minutes = float(hours_minutes[1])
            seconds_milliseconds = hours_minutes[2].split(".",-1)
            seconds = float(seconds_milliseconds[0])
            milliseconds = float(seconds_milliseconds[1])
            totalTimeMilli = totalTimeMilli + ((hours*60000*60)+(60000*minutes)+(1000*seconds)+(milliseconds))
                
            #print hours
            #print minutes
            #print seconds
            #print milliseconds
            #print totalTimeMilli
            
        #cut audio
        #calculate new startTimeMilli and EndTimeMilli of clip
        startTimeMilli = startTimeMilli-offset
        endTimeMilli = endTimeMilli-offset
                
        #print 'In clip start time is:', startTimeMilli
        #print 'In clip end time is:', endTimeMilli
            
        cutAudio = audioFile[startTimeMilli:endTimeMilli]
        print("cut Audiofile")
    
        #get ID
        nameParse = AudioClips.split("L",-1)
        #print nameParse
        ID = nameParse[0]
        #print ID
            
        #Differentiate between buzz and minibuzz and Make event audio filename 
        if (callType == "buzz\n"):        
            buzzCounter = buzzCounter+1
            print'Buzz detected!\n'
            cutAudioFilename = ID+'_event'+str(eventCounter)+'_buzz'+str(buzzCounter)+'.wav'
            #save event audio
            #print cutAudioFilename
            cutAudio.export(cutAudioFilename, format="wav")
            #print("Exported!")
                
        elif (callType == "minibuzz\n"): 
            minibuzzCounter = minibuzzCounter+1
            print'Minibuzz detected!\n'
            cutAudioFilename = ID+'_event'+str(eventCounter)+'_minibuzz'+str(minibuzzCounter)+'.wav'
            #save event audio
            #print cutAudioFilename
            cutAudio.export(cutAudioFilename, format="wav")
            #print("Exported!")
    
    auditFile.close()
    audioClipsFile.close()
    print(".txt files closed!")

#segmentAudio('D:\VA_IndStudy\Testing', 'testLog.txt', 'testAud.txt')
    
#segmentAudio('D:\VA_IndStudy\Data', 'gm185bLog.txt', 'gm10_185baud.txt')

#segmentAudio('D:\VA_IndStudy\Data', 'gm187bLog.txt', 'gm10_187baud.txt')
#segmentAudio('F:\VA_IndStudy\Data', 'gm208aLog.txt', 'gm10_208aaud.txt')
#segmentAudio('F:\VA_IndStudy\Data', 'gm209cLog.txt', 'gm10_209caud.txt')
#segmentAudio('F:\VA_IndStudy\Data', 'gm266aLog.txt', 'gm10_266aaud.txt')
#segmentAudio('F:\VA_IndStudy\Data', 'gm267aLog.txt', 'gm10_267aaud.txt')

###2008
#segmentAudio('D:\Pilotwhales2008\data\gm08\gm08_143a\Use', 'gm143aLog.txt', 'gm08_143aaud.txt')
#segmentAudio('D:\Pilotwhales2008\data\gm08\gm08_143b\Use', 'gm143bLog.txt', 'gm08_143baud.txt')
#segmentAudio('D:\Pilotwhales2008\data\gm08\gm08_147a\Use', 'gm147aLog.txt', 'gm08_147aaud.txt')
#segmentAudio('D:\Pilotwhales2008\data\gm08\gm08_151a\Use', 'gm151aLog.txt', 'gm08_151aaud.txt')
#segmentAudio('D:\Pilotwhales2008\data\gm08\gm08_151b\Use', 'gm151bLog.txt', 'gm08_151baud.txt')

###2011

segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_147a\use', 'gm147aLog.txt', 'gm11_147aaud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_148a\use', 'gm148aLog.txt', 'gm11_148aaud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_149b\use', 'gm149bLog.txt', 'gm11_149baud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_149c\use', 'gm149cLog.txt', 'gm11_149caud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_150b\use', 'gm150bLog.txt', 'gm11_150baud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_155a\use', 'gm155aLog.txt', 'gm11_155aaud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_156a\use', 'gm156aLog.txt', 'gm11_156aaud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_158b\use', 'gm158bLog.txt', 'gm11_158baud.txt')
segmentAudio('D:\Pilotwhales2011\data\gm11\gm11_165a\use', 'gm165aLog.txt', 'gm11_165aaud.txt')
