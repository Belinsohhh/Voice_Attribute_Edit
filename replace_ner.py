from openai import OpenAI
import pandas as pd
import json

file = input("Enter parquet file name:")

client = OpenAI(api_key= "api key")
GPT_MODEL = "gpt-4o-mini"

data = pd.read_parquet('voxpopuli/voxpopuli/' + file, engine='pyarrow')

normalized_text = data["normalized_text"]
normalized_combined_ner = data["normalized_combined_ner"]

function_metrics_gpt = [
{
    "name": "PresentNERReplacement",
    "description": "Evaluate the following details for sentence provided by user",
    "parameters": 
    {
        "type": "object",
        "properties": 
        {
            "Sentence": 
            {
                "type": "array",
                "description": "description of sentence in array",
                "items": 
                {
                    "type": "object",
                    "properties": 
                    {
                        "Original Sentence": 
                        {
                            "type": "string",
                            "description": "sentence provided by user"
                        },
                        "NER Instructions": 
                        {
                            "type": "string",
                            "description": "ner instructions given by user "
                        },
                        "Replaced Sentence": 
                        {
                            "type": "string",
                            "description": "This is the sentence that has its ner words all replaced."
                        },
                    }
                }
            }
        }
    }
}]

def replace_ner_gpt_edited(query: str, ner:str, model: str = GPT_MODEL,) -> str:
    messages = [
        {"role": "system", "content": '''You are a Named Entity Recognition (NER) expert. You will replace the sensitive NER words in the sentence provided by user with a completely dissimilar replacement and output the original sentence, ner instructions and replaced sentence in json format like the following examples in delimitered by XML tags:
        <output>
        Example 1:
        {
            "Original Sentence": "secondly conclusions from the commission communication entitled towards world class clusters in the european union implementing the broad based innovation strategy",
            "NER Instructions": "Replace the 'PLACE' type word, starting from character 100 to character 113, with a dissimilar word. Replace the 'QUANT' type word, starting from character 0 to character 7, with a dissimilar word.",
            "Replaced Sentence": "Next, conclusions from the commission communication entitled towards world-class clusters across the continent implementing the comprehensive innovation strategy." 
        }
         
        Example 2:
        {
            "Original Sentence": "along with eighty four other meps i have tabled a plenary amendment which i really hope you can all support.",
            "NER Instructions": "Replace the 'QUANT' type word, starting from character 11 to character 21, with a dissimilar word.",
            "Replaced Sentence": "Along with a substantial group of other MEPs, I have tabled a plenary amendment which I really hope you can all support."
        }
        
         Example 3:
        {
            "Original Sentence": "in this situation we have to see reflection political responsibility and more political dialogue.",
            "NER Instructions": "",
            "Replaced Sentence": "in this situation we have to see reflection political responsibility and more political dialogue."
        }
        
        Derive the answer step by step:
        1. Obtain the sentence from user
        2. Replace each word in the original sentence, as specified by the users NER instructions, with a dissimilar word. If no NER instructions is given, DO NOT do any replacements.
        4. Input original sentence, NER instructions and replaced sentence into the json format required. 

        Take a deep breath and think carefully, then only reply with the final json format without the intermediate steps.'''},
        {"role": "user", "content": "give me a replaced sentence based on the original sentence: " + query + ", and NER instructions:" + ner},
    ]

    response = client.chat.completions.create(
        model=model,
        messages = messages,
        response_format = {"type": "json_object"},
        functions = function_metrics_gpt,
        function_call = {"name": "PresentNERReplacement"},
        temperature=0
    )
    
    argument = response.choices[0].message.function_call.arguments
    json_obj = json.loads(argument)
    return json_obj["Sentence"]

replaced_output = []
for idx, each_text in enumerate(normalized_text):
    ner_type_list = normalized_combined_ner[idx].get("type")
    ner_start_list = normalized_combined_ner[idx].get("start")
    ner_length_list = normalized_combined_ner[idx].get("length")
    replace_command = ""
    if len(ner_type_list) != 0:
        for itr, each_replace in enumerate(ner_type_list):
            start = ner_start_list[itr]
            end = ner_start_list[itr] + ner_length_list[itr] - 1
            replace_command += f'Replace the {each_replace} ner type word, starting from character {start} to character {end}, with a dissimilar word. '
        output = replace_ner_gpt_edited(each_text, replace_command)
        replaced_output.append(output[0])

with open('ner_output.json', 'w') as f:
    json.dump(replaced_output, f, indent=4)