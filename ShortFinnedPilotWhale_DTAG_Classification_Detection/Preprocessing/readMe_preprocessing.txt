Order of preprocessing: 

Need: Log file, audit file, .wav DTAG recordings
1. Run audioParse to get buzz and minibuzz clips
2. Run high pass filter on all of sound clips
3. Compute STFT of the filtered audio clips

--> STFTs and labels serve as input for the machine learning model