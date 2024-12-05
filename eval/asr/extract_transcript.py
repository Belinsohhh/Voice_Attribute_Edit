import pandas as pd

# Replace the file path with your actual file path
file_path = '/home/xiaoxiao/Voice_Attribute_Edit/data/slue-voxpopuli/slue-voxpopuli_dev.tsv'
output_file_path = 'key.txt'
# Read the TSV file, specifying the columns we are interested in
df = pd.read_csv(file_path, sep='\t', usecols=['id', 'raw_text'])

with open(output_file_path, 'w') as file:
    # Iterate over the dataframe rows
    for index, row in df.iterrows():
        # Write id and raw_text to the file
        file.write(f"{row['id']} {row['raw_text']}\n")

print(f"Data has been written to {output_file_path}")
