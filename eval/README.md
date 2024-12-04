## Installation
Package
```bash
pip install speechbrain
```
Clone slue-toolkit for benchmark datasets
```bash
git clone https://github.com/asappresearch/slue-toolkit.git
```

## eval ASR
1. convert ogg to wav: change `$input_folder` in `eval/asr/00_convert2wav.sh`
   
   `bash eval/asr/00_convert2wav.sh`
   
2. compute wer for orignal and generated speech: change `$dir` `$json_file` in `eval/asr/01_predict.sh`

change file directories as necessary. Current compilation is done at main folder level

   `bash eval/asr/01_predict.sh`

## eval ASV
change 
`label file`: /root/slue-toolkit/data/slue-voxpopuli/slue-voxpopuli_test_blind.tsv

`original speech`: /root/Voice_Attribute_Edit/eval/asr/ori_wav16k16bit

`generated speech`: /root/Voice_Attribute_Edit/Random_Speaker
in `eval/asv/run.sh`

`bash eval/asv/run.sh`

