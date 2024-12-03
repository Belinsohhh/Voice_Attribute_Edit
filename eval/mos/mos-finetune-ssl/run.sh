#!/bin/bash
data_dir=../resample_norm/wav_norm/
dset=

python run_inference_for_challenge.py --datadir $data_dir/$dset 
echo The individual scores are save in answer_${dset}.txt


echo  The mean of answer_$dset.txt is 
awk '{ total += $2; count++ } END { print total/count }' answer_${dset}.txt 
