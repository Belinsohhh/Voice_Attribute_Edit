#!/bin/bash

stage=1

# Stage 1: Predict original audio
if [ "$stage" -le 1 ]; then
    dir="/root/Voice_Attribute_Edit/eval/asr/ori_wav16k16bit"
    outname="ori_predict.txt"

    # Generate a list of WAV files
    find "$dir" -name "*.wav" -type f > wav.lst

    # Run prediction
    python main.py wav.lst "$outname"

    json_file="/root/Voice_Attribute_Edit/ner_replaced_new_edited.json"

    # Output file
    output_file="ori_key.txt"

    # Process the JSON file to extract id and Original Sentence, replacing ":" in id
    cat "$json_file" |
    grep -E '"id"|"Original Sentence"' | # Extract lines with "id" or "Original Sentence"
    sed -E 's/.*"id": "(.*)",/\1/;s/.*"Original Sentence": "(.*)"/\1/' | # Extract values
    # sed 's/://g' | # Replace ":" with an empty string in the id
    awk 'NR % 2 == 1 {id=$0; next} {print id, $0}' > "$output_file" # Combine "id" and "Original Sentence"


    echo "Data has been saved to $output_file"

    # Compute WER for original audio
    python compute_wer.py --mode present "$output_file" "$outname"
fi

# Stage 2: Predict generated audio
if [ "$stage" -le 2 ]; then
    dir="../../Random_Speaker"
    outname="Random_Speaker_predict.txt"

    # Generate a list of WAV files
    find "$dir" -name "*.wav" -type f > wav.lst

    # Run prediction
    python main.py wav.lst "$outname"

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
    python compute_wer.py --mode present "$output_file" "$outname"
fi

