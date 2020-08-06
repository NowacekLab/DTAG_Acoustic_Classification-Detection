Experiment in detection using auditok library to isolate events to feed into classifier

*Disclaimer: overall, did not preform too well, would not recommend continuing*

0. Install auditok library and other dependencies 

1. Run high pass filter on whole DTAG .wav file (HighPassFilterLoop4_bigfiles.m)

2. Run the auditok library on DTAG .wav file 
	Run auditok in commend line with desired parameters (see DetectionsRun.PNG)
	Attempted to automate with bash script (ex: F_gm10_188a.sh)

3. Plot results and calculate statistics to evaluate detector success (PlotDetectedAndAudit_waveform_loop.py)

Side note: "downsample_comparision.py" used to experiment with doensampling methods to graph large .wav files
Librosa library was found to be best

