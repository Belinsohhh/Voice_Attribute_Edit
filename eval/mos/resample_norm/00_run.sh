#!/bin/sh
# This script downsample the waveform to 16kHz and normalize the waveform amplitude
# Usage:
#  1. Modify SOURCE_DIR, TARGET_DIR
#  2. Specify the path of SOX and SV56 in sub_down_sample.sh and sub_sv56.sh
#     SOX can be downloaded from https://sourceforge.net/projects/sox/
#     SV56 can be downloaded from https://www.itu.int/rec/T-REC-G.191-201901-I/en 
#  3. sh 00_batch.sh
# Requirement:
#  sox, sv56


# Directory of input waveforms
flag=
INPUT_WAV_DIR=/home/xiaoxiao/Voice_Attribute_Edit/eval/asr/ori_wav16k16bit/$flag

OUTPUT_DIR=./wav_norm/$flag
# Directory to store the processed waveforms
OUTPUT_WAV_DIR=${OUTPUT_DIR}/wav
# Sampling rate to be used
SAMP=16000

mkdir -p ${OUTPUT_WAV_DIR}
#get file list
find ${INPUT_WAV_DIR} -name "*.wav" -type f -o -type l > __tmp_file.lst

#resample-norm

#cat __tmp_file.lst | parallel -j 5  sh sub_16khz16bit.sh {} ${OUTPUT_WAV_DIR}/{/} ${SAMP}
cat __tmp_file.lst | parallel -j 5 bash sub_sv56.sh ${OUTPUT_WAV_DIR}/{/} ${OUTPUT_WAV_DIR}/{/}

#create sets for MOS
mkdir -p ${OUTPUT_DIR}/sets
find ${OUTPUT_WAV_DIR} -type f -printf "%f\n" > ${OUTPUT_DIR}/sets/temp.txt
sed 's/$/,1.0/g' ${OUTPUT_DIR}/sets/temp.txt > ${OUTPUT_DIR}/sets/val_mos_list.txt
rm ${OUTPUT_DIR}/sets/temp.txt
