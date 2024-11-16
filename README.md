# Privacy in Generated Synthetic Voice
This folder contains 2 separate tasks:
1. Generation of Synthetic Audio Clips
2. Replacement of Sensitive Words (NER) from Transcripts

## Table of Contents
- [Installation](#installation)
- [Generation Usage](#generation)
- [NER Replacement Usage](#replacement)

## Installation
1. Clone the repository:
```bash
 git clone https://github.com/yourusername/Voice_Attribute.git
```

2. Unzip voxpopuli.zip

3. Create a virtualenv then:
```bash
 pip install -r requirements.txt
```

## Generation Usage
These packages generate synthetic audio files reading random transcripts from LibriSpeech ASR corpus. For each of the 34 specific speakers, 10 utterances is generated. For random speakers, 156 random speakers are generated using a random combination of gender, pitch, accent. And each of the random speakers are used to generate 10 utterances with varying pitch modulation, speaking rate, and channel condition. 

1. Run for specific speaker generation: 
```bash
 python generate_specific_speaker.py
```
Output files are saved in specific_speaker:
- Sound files is saved in .wav format 
- Corresponding Transcript saved in .txt format

2. Run for random speaker generation:
```bash
 python generate_random_speaker.py
```
Output files are saved in random_speaker:
- Sound files is saved in .wav format 
- Corresponding Transcript saved in .txt format

## NER Replacement Usage
This package is meant for using GPT to replace NER words according to instructions to increase privacy around transcripts used. The model used is: gpt-4o-mini. This programme is able to extract from .parquet files given that the column headers are "normalized_text" and "normalized_combined_ner" as modelled after SLUE-VoxPopuli transcripts..
1. Run for NER word replacement:
```bash
 python replace_ner.py
```

2. Enter .parquet file for analysis:
E.g. Enter parquet file name: test-00000-of-00001.parquet

Output is saved in ner_output.json.