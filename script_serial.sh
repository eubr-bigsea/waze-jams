STARTTIME=$(date +%s)


octave prepare.m sample_train.txt

for N in `seq 1 20`
do
	octave runGP.m config.txt $N
done

ENDTIME=$(date +%s)

echo "It took $(($ENDTIME - $STARTTIME)) seconds"
