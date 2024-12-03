1. resample and norm the audios

cd resample_norm 

bash 00_run.sh

2. Predict MOS

https://github.com/Belinsohhh/Voice_Attribute_Edit/releases/download/untagged-e230b770460e7dbd9d8b

download ckpt_w2vsmall and copy to mos-finetune-ssl/pretrained

download wav2vec_small.pt and copy to mos-finetune-ssl/fairseq/

cd mos-finetune-ssl

bash run.sh
