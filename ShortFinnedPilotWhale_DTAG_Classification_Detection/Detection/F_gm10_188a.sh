#run auditok detections in command line with ditterent threshold values

#!/bin/bash
set -e
#cd F:/Pilotwhales2010/data/m10/gm10_188a/Detection_Test

#/c/Users/jpan2/Anaconda2/Scripts/conda.exe activate tensoflow_env
for i in {30..34..2};
do
    echo "threshold value $i"
	auditok -e $i -i ./F_gm188a01.wav -n 0.1 -m 20 -s 0.5>"F_gm188a01_$i.txt"
	echo "processed value with threshold value $i"
	#echo "going to graph threshold value $i results"
	#python plotDetectedSounds_v3.py doorKnock.wav doorKnock_$i.txt audit_doorKnock.txt
	#echo "graphed threshold $i results"
done