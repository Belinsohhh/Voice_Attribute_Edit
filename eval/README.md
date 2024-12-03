## Installation

```
pip install speechbrain
```


## eval ASR
1. convert ogg to wav: change `$input_folder` in `eval/asr/00_convert2wav.sh`
   
   `bash eval/asr/00_convert2wav.sh`
   
2. compute wer for orignal and generated speech: change `$dir` `$json_file` in `eval/asr/01_predict.sh`

   `bash eval/asr/01_predict.sh`

## eval ASV
change 
`label file`: /root/slue-toolkit/data/slue-voxpopuli/slue-voxpopuli_test_blind.tsv

`original speech`: /root/Voice_Attribute_Edit/eval/asr/ori_wav16k16bit

`generated speech`: /root/Voice_Attribute_Edit/Random_Speaker
in `eval/asv/run.sh`

`bash eval/asv/run.sh`

## eval MOS

### Step 1: Resample and Normalize the Audios

cd mos/resample_norm

1. resample and norm the audios

cd `mos/resample_norm` 

`bash install_sv56.sh`

`bash 00_run.sh`

2. Predict MOS

click `Releases->Assets`

download `ckpt_w2vsmall` and copy to `mos/mos-finetune-ssl/pretrained`

download `wav2vec_small.pt` and copy to `mos/mos-finetune-ssl/fairseq/`

cd `mos/mos-finetune-ssl`

`bash INSTALL.sh`

change `INPUT_WAV_DIR` in `00_run.sh` https://github.com/Belinsohhh/Voice_Attribute_Edit/blob/main/eval/mos/resample_norm/00_run.sh#L15

`bash 00_run.sh`

