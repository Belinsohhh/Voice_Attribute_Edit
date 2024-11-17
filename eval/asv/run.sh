mkdir -p exp

find /root/Voice_Attribute_Edit/Random_Speaker -type f -name *wav > done.lst
python compute_em.py /root/slue-toolkit/data/slue-voxpopuli/slue-voxpopuli_test_blind.tsv \
	/root/Voice_Attribute_Edit/eval/asr/ori_wav16k16bit \
	/root/Voice_Attribute_Edit/Random_Speaker exp \
        done.lst
