import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device": device} )
def extract_vector(audio):
        signal, fs =torchaudio.load(audio)
        return classifier.encode_batch(wavs=signal).squeeze(0)


def extract_vector_pair(audio):
        signal, fs =torchaudio.load(audio)
        length = signal.shape[-1]
        signal1 = signal[:, :int(length/2)]
        signal2 = signal[:, int(length/2):]

        return classifier.encode_batch(wavs=signal1).squeeze(0), classifier.encode_batch(wavs=signal2).squeeze(0)



# functions to get EERs
def compute_det_curve(target_scores, nontarget_scores):
    """ 
    frr, far, thr = comcompute_eercurve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores 

    output
    ------
      frr:   np.array, FRR, (#N, ), where #N is total number of scores + 1
      far:   np.array, FAR, (#N, ), where #N is total number of scores + 1
      thr:   np.array, threshold, (#N, )
    
    """
    
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), 
                             np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size - 
                            (np.arange(1, n_scores + 1) - tar_trial_sums))

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums/target_scores.size))
    # false rejection rates
    far = np.concatenate((np.atleast_1d(1), 
                          nontarget_trial_sums / nontarget_scores.size))  
    # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), 
                                 all_scores[indices]))  
    # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    eer, eer_threshold = compute_eer(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, or list of np.array, target trial scores
      nontarget_scores: np.array, or list of np.array, nontarget trial scores 

    output
    ------
      eer:            float, EER 
      eer_threshold:  float, threshold corresponding to EER
    
    """
    if type(target_scores) is list and type(nontarget_scores) is list:
        frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    else:
        frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    
    # find the operation point for EER
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)

    # compute EER
    eer = np.mean((frr[min_index], far[min_index]))    
    return eer, thr[min_index]


def compute_far(scores, threshold):
    """
    far = compute_far(scores, threshold)
    
    Compute false acceptance rate
    
    input
    -----
      scores:    np.array, or list of np.array, trial scores
      threshold: float, threshold for decision

    output
    ------
      far:            float, false acceptance rate
    """
    return np.sum(scores > threshold) / scores.size


def gen_pos_neg(file_path, done):
  # Read the TSV file, specifying the columns we are interested in
  df = pd.read_csv(file_path, sep='\t', usecols=['id', 'speaker_id'])

  # Remove rows where speaker_id is None
  df = df[df['speaker_id'].notna()]

  # Initialize lists to store positive and negative pairs
  positive_pairs = []
  negative_pairs = []

  # Convert the DataFrame to a list of tuples for easier iteration
  records = df.to_records(index=False)
  record_list_temp = list(records)
  record_list = []
  for record in record_list_temp:
      temp = record[0].replace(':','')
      if temp in done:
          record_list.append(record)

  # Generate pairs
  for i in range(len(record_list)):
      for j in range(i + 1, len(record_list)):
          id1, speaker_id1 = record_list[i]
          id2, speaker_id2 = record_list[j]
          
          if speaker_id1 == speaker_id2:
              positive_pairs.append((id1, id2))
          else:
              negative_pairs.append((id1, id2))
  
  # Write the pairs to output files
  with open('trials.txt', 'w') as file:
      for pair in positive_pairs:
          file.write(f"{pair[0]} {pair[1]} 1\n")
      for pair in negative_pairs:
          file.write(f"{pair[0]} {pair[1]} 0\n")

  return  positive_pairs, negative_pairs


done = sys.argv[5]
done_list = []
for line in open(done):
    done_list.append(line.strip().split('/')[-1].split('.')[0])
input_key = sys.argv[1]
positive_pairs, negative_pairs = gen_pos_neg(input_key,done_list)
# Choose a subset
positive_pairs = positive_pairs[:]
negative_pairs = negative_pairs[:]

ori_dir = sys.argv[2]
anon_dir = sys.argv[3]
out = sys.argv[4]
neg = []
pos = []
oa = []

#print(len(positive_pairs), len(negative_pairs))

for pos_pair in positive_pairs:
  id1 = pos_pair[0]
  id2 = pos_pair[1]

  line_spks = []
  wav1 = ori_dir + '/' + id1 + '.wav'
  wav2 = ori_dir + '/' + id2 + '.wav'
  wav3 = anon_dir + '/' + id2.replace(':','') + '.wav'

  vector1 = extract_vector(wav1).cpu().numpy()
  vector2 = extract_vector(wav2).cpu().numpy()
  vector3 = extract_vector(wav3).cpu().numpy()

  distance_array = cosine_distances(vector1, vector2)[0, 0]
  cosine_distance = 1 - float(distance_array)  # Assuming distance_array is a single value
  pos.append(cosine_distance)

  dis2 = cosine_distances(vector1, vector3)[0, 0]
  cos2 = 1 - float(dis2)  # Assuming distance_array is a single value
  oa.append(cos2)


for neg_pair in negative_pairs:
  id1 = neg_pair[0]
  id2 = neg_pair[1]
  line_spks = []
  wav1 = ori_dir + '/' + id1 + '.wav'
  wav2 = ori_dir + '/' + id2 + '.wav'
  vector1 = extract_vector(wav1).cpu().numpy()
  vector2 = extract_vector(wav2).cpu().numpy()
  distance_array = cosine_distances(vector1, vector2)[0, 0]
  cosine_distance = 1 - float(distance_array)  # Assuming distance_array is a single value
  neg.append(cosine_distance)



pos = np.array(pos)
neg = np.array(neg)
oa = np.array(oa)

np.save('%s/pos.npy'%out, pos)
np.save('%s/neg.npy'%out, neg)
np.save('%s/oa.npy'%out, oa)

pos = np.load('%s/pos.npy'%out)
neg = np.load('%s/neg.npy'%out)
oa = np.load('%s/oa.npy'%out)

eer, threshold = compute_eer(oa, neg)
print("%f%"%eer*100)
