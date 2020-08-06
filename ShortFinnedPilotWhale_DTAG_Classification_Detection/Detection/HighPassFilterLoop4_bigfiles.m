%% Code used to apply high pass Butterwork filter filter 
%%(cutoff frequency 1,000Hz) to whole DTAG audio recordings

%2008
%cd 'D:\Pilotwhales2008\data\gm08\gm08_147a\Use'
%cd 'D:\Pilotwhales2008\data\gm08\gm08_143a\Use'
%cd 'D:\Pilotwhales2008\data\gm08\gm08_143b\Use'
%cd 'D:\Pilotwhales2008\data\gm08\gm08_151a\Use'
%cd 'D:\Pilotwhales2008\data\gm08\gm08_151b\Use'

%2011
%cd 'D:\Pilotwhales2011\data\gm11\gm11_147a\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_148a\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_149b\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_149c\use' 
%cd 'D:\Pilotwhales2011\data\gm11\gm11_150b\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_155a\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_156a\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_158b\use'
%cd 'D:\Pilotwhales2011\data\gm11\gm11_165a\use'
clear; 
cd 'D:\Pilotwhales2010\data\gm10\gm10_188a'
%Make filter
%Apply high pass filter
fc = 1000; % Cut off frequency
Fs = 96000;
%[b,a] = butter(n,Wn,ftype) 
%designs a lowpass, highpass, bandpass, 
%or bandstop Butterworth filter, depending on the value of ftype 
%and the number of elements of Wn. 
%The resulting bandpass and bandstop designs are of order 2n.
[b,a] = butter(6,fc/(Fs/2),'high'); % Butterworth filter of order 6
freqz(b,a)
%Loop through all .wav files in folder
files = dir('*.wav');
for file = files'
     name = getfield(file,'name');
     fprintf('Now processing file: %s\n', name)
     [y,Fs] = audioread(name);
     %fprintf('%i\n', Fs)

     %apply filter
     yFiltered = filter(b,a,y); % Will be the filtered signal

%     %Normalize to prevent clipping?-must be in range [-1,1)-doesn't seem to filter as well 
%     %yFiltered = yFiltered./(max(abs(yFiltered)));
     
     clear y;
     fprintf('deleted original audio data\n')
     
     %write filtered sound to .wav
     filename = strcat('F_',name);
     %fprintf('%s\n', filename)

     audiowrite(filename,yFiltered,Fs)
end
