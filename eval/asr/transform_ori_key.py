import json
with open("original_transcripts.json", "r") as f:
    data = json.load(f)


with open("ori_key.txt", "w") as f:
    for i in data:
        f.write(str(i.get("id")) + " " + str(i.get("Original Sentence")))
        f.write("\n")