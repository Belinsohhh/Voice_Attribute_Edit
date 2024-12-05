#!/bin/bash

stage=1

# Stage 1: Predict original audio
if [ "$stage" -le 1 ]; then
    dir="ori_wav16k16bit"
    outname="ori_predict.txt"
    outputfile="key.txt"
    # Generate a list of WAV files
    find "$dir" -name "*.wav" -type f > wav.lst
    cat wav.lst
    # Run prediction
    python eval/asr/main.py wav.lst "$outname"

    json_file="original_transcripts.json"

    # Output file
    output_file="ori_key.txt"

    # Process the JSON file to extract id and Original Sentence
    python eval/asr/transform_ori_key.py 

    echo "Data has been saved to $output_file"

    # Compute WER for original audio
    python eval/asr/compute_wer.py --mode present "$output_file" "$outname"
fi
exit 0
# Stage 2: Predict generated audio
if [ "$stage" -le 2 ]; then
    dir="Specific_Speaker" #replace with Random_Speaker to get Random Speaker Predictions
    outname="Specific_Speaker_predict.txt" #replace with Random_Speaker to get Random Speaker Predictions

    # Generate a list of WAV files
    find "$dir" -name "*.wav" -type f > wav.lst

    # Run prediction
    python eval/asr/main.py wav.lst "$outname"

    # Create or clear the output file
    output_file="key.txt"
    > "$output_file"

    # Process .txt files in the folder
    for file in "$dir"/*.txt; do
        # Read the content of the file
        content=$(cat "$file")
        # Get the filename (without path)
        filename=$(basename "$file" .txt)
        # Append the filename and content to the output file
        echo "$filename $content" >> "$output_file"
    done

    # Compute WER for generated audio
    python eval/asr/compute_wer.py --mode present "$output_file" "$outname"
fi

