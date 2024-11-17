#!/bin/bash

input_folder="/root/slue-toolkit/data/slue-voxpopuli/test"
output_folder="ori_wav16k16bit"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Loop through all .wav files in the input folder
for file in "$input_folder"/*.ogg; do
    # Get the base filename without extension
    base_name=$(basename "$file" .ogg)
    
    # Convert the file
    ffmpeg -i "$file" -ar 16000 -ac 1 -sample_fmt s16 "$output_folder/${base_name}.wav"
    
done

