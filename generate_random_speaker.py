# !pip install torch
# !pip install git+https://github.com/huggingface/parler-tts.git
# !pip install transformers
# !pip install soundfile

import random
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import json
import itertools

#transcript and og file id
with open('ner_replaced_transcripts.json', 'r') as file:
    data = json.load(file)

transcripts = []
id = []
for each in data:
    transcripts.append(each.get("Replaced Sentence"))
    id.append(each.get("id"))

#gender
my_file = open("attributes/gender.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
gender = data.split("\n") 
my_file.close() 

#accent
my_file = open("attributes/accents.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
accents = data.split("\n") 
my_file.close() 

#pitch
my_file = open("attributes/pitch.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
pitch = data.split("\n") 
my_file.close() 

#modulation
my_file = open("attributes/modulation.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
modulation = data.split("\n") 
my_file.close() 

#rate
my_file = open("attributes/rate.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
rate = data.split("\n") 
my_file.close() 

#channel
my_file = open("attributes/channel conditions.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
channel = data.split("\n") 
my_file.close() 

#distance
my_file = open("attributes/distance.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
distance = data.split("\n") 
my_file.close() 

#recording
my_file = open("attributes/recording.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
recording = data.split("\n") 
my_file.close() 

def cross_combinations(list1, list2):
    listing = list(itertools.product(list1, list2))
    new_list = []
    for i in listing:
        new_list.append(list(i))
    return new_list

identity = cross_combinations(cross_combinations(gender, accents), pitch)

def generate_random_env(channel, distance, recording):
    if channel!="" and distance !="" and recording !="":
        random_val = random.choice([1,0])
        if random_val == 0:
            recording = ""
        else:
            distance = ""

    if channel=="" and distance =="" and recording =="":
        environment = ""
    elif channel=="":
        if distance == "":
            environment = f"The speaker's voice is {distance}, and the {recording}."
        else:
            environment = f"The {recording}."
    elif recording == "":
        if distance == "":
            environment = f"The speaker's voice is {distance}, and {channel}."
        else:
            environment = f"The speaker's voice is {channel}."
    else:
        environment = ""

    return environment

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

meta_data = []
for num, random_speaker in enumerate(identity):
    random_gender = random_speaker[0][0]
    random_accent = random_speaker[0][1]
    random_pitch = random_speaker[1]
    for i in range(10):
        meta = {}
        random_channel = random.choice(channel)
        random_distance = random.choice(distance)
        random_recording = random.choice(recording)
        random_environment = generate_random_env(random_channel, random_distance, random_recording)
        random_modulation = random.choice(modulation)
        random_rate = random.choice(rate)
        prompt = random.choice(transcripts)
        index = transcripts.index(prompt)
        actual_id = str(id[index])
        description = f"A {random_gender} voice in a {random_accent} accent reads a book {random_rate} with a {random_pitch} {random_modulation} voice. {random_environment}"

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        title = f"Random_Speaker/{num}-{random_gender}-{random_accent}-{random_rate}-{random_pitch}-{random_modulation}-{random_distance}-{random_channel}-{random_recording}"

        meta["id"] = actual_id
        meta["Description Summary"] = title
        meta["TTS Description"] = description
        meta["TTS Prompt"] = prompt
        meta_data.append(meta)

        sf.write("Random_Speaker/" + str(actual_id.replace(":", "")) +".wav", audio_arr, model.config.sampling_rate)
        f = open("Random_Speaker/" + str(actual_id.replace(":", "")) + ".txt", "a")
        f.write(prompt)
        f.close()
        
with open('generated_meta_info_random_speaker.json', 'w') as f:
    json.dump(meta_data, f, indent=4)