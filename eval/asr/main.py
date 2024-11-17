import torch
import torchaudio
from speechbrain.pretrained import EncoderDecoderASR
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
import librosa
import soundfile as sf

class MyDataset(Dataset):
    def __init__(self, wavlist):
        f = open(wavlist, 'r')
        data = []
        for line in f:
            filename = line.strip().split(' ')[-1]
            wavpath = filename
            wavname = wavpath.split('/')[-1].split('.')[0]
            wav_old, original_sr  = torchaudio.load(wavpath)
            if original_sr != 16000:
                # Convert the tensor to a numpy array and resample using librosa
                wav_resampled = torch.tensor(librosa.resample(wav_old.squeeze().numpy(), orig_sr=original_sr, target_sr=16000))
            else:
                wav_resampled = wav_old.squeeze()

            # Get the length of the resampled audio
            wav_len = len(wav_resampled)

            # Append to the data list
            data.append((wavname, wav_resampled, wav_len))
            
        # Sort the data based on audio length
        self.data = sorted(data, key=lambda x: x[2], reverse=True)

    def __getitem__(self, idx):
        wavname, wav, wav_len = self.data[idx]
        return wav, wavname, wav_len

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):  ## make them all the same length with zero padding
        wavs, wavnames, wav_lens = zip(*batch)
        wavs = list(wavs)
        batch_wav = pad_sequence(wavs, batch_first=True, padding_value=0.0)
        lens = torch.Tensor(wav_lens) / batch_wav.shape[1]


        return wavnames, batch_wav, lens

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from speechbrain.inference.ASR import EncoderASR

asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")

#asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech",run_opts={"device":device})
#asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en", savedir="pretrained_models/asr-wav2vec2-commonvoice-en", run_opts={"device": device})
devlist = sys.argv[1]
devset = MyDataset(devlist)
devloader = DataLoader(devset, batch_size=12, shuffle=False, num_workers=0, collate_fn=devset.collate_fn)

# Specify the output file path
output_file_path = sys.argv[-1]

with torch.no_grad():
# Open the output file for writing
    print(output_file_path)
    with open(output_file_path, 'w') as outfile:
        for i, data in enumerate(devloader, 0):
            filenames,inputs, lengths = data
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            predicts, _ = asr_model.transcribe_batch(inputs, lengths)
        
            # Iterate over filenames and predicts, and write them line by line
            for filename, prediction in zip(filenames, predicts):
                # Write the filenames and predictions to the output file
                #print(filename,prediction)
                line = f"{filename} {prediction}\n"
                outfile.write(line)

# Print a message ind:icating the process is complete
print(f"Results written to {output_file_path}")
