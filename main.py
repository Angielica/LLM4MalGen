import os
import json
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm

from utility.utils import *


def main(fname):
    with open(fname) as f:
        params = json.load(f)

    if not os.path.exists(os.path.join('./data/', 'CSV')):
        os.mkdir(os.path.join('./data', 'CSV'))
        os.mkdir(os.path.join('./data', 'CSV', 'Benign assembly'))
        os.mkdir(os.path.join('./data', 'CSV', 'Virus assembly'))
        os.mkdir(os.path.join('./data', 'CSV', 'Virus assembly', 'Winwebsec'))

    # Convert exe file into csv file
    if params['generate_csv']:
        benign_assembly = os.listdir('./data/Benign assembly')
        for file in tqdm(benign_assembly, total=len(benign_assembly), desc='Benign assembly - CSV generation'):
            if 'exe.asm' in file:
                df_path = './data/CSV/Benign assembly/'+ file.split('.exe.asm')[0] + '.csv'
                exe_path = os.path.join('./data/Benign assembly', file)
                generate_csv_for_each_sample(exe_path, df_path)

        winwebsec = os.listdir('./data/Virus assembly/Winwebsec')
        for file in tqdm(winwebsec, total=len(winwebsec), desc='Winwebsec - CSV generation'):
            if 'exe.asm' in file:
                df_path = './data/CSV/Virus assembly/Winwebsec/' + file.split('.exe.asm')[0] + '.csv'
                exe_path = os.path.join('./data/Virus assembly/Winwebsec', file)
                generate_csv_for_each_sample(exe_path, df_path)

    if params['generate_embeddings']:
        section_to_analyze = params['section_to_analyze']

        language_model_id = params['language_model_id']
        model = SentenceTransformer(language_model_id)

        # Generate embedding for benign data
        new_data = []
        benign_csv = os.listdir('./data/CSV/Benign assembly')
        for file in tqdm(benign_csv, total=len(benign_csv), desc='Benign - Embedding generation'):
            if 'csv' in file:
                df_path = './data/CSV/Benign assembly/' + file
                df = pd.read_csv(df_path)
                df = df[df['Section'] == section_to_analyze]
                df = df.dropna()

                instruction = df['Instruction'].values.tolist()
                instruction = ' '.join(instruction)

                embedding = generate_embeddings(model, instruction)
                new_data.append([file.split('.csv')[0], instruction, embedding])

        new_data_df = pd.DataFrame(new_data, columns=['Sample Name', 'Instruction', 'Embedding'])
        new_data_df.to_parquet('./data/benign.parquet')

        # Generate embedding for winwebsec
        new_data = []
        winwebsec_csv = os.listdir('./data/CSV/Virus assembly/Winwebsec')
        for file in tqdm(winwebsec_csv, total=len(winwebsec_csv), desc='Winwebsec - Embedding generation'):
            if 'csv' in file:
                df_path = './data/CSV/Virus assembly/Winwebsec/' + file
                df = pd.read_csv(df_path)
                df = df[df['Section'] == section_to_analyze]
                df = df.dropna()

                instruction = df['Instruction'].values.tolist()
                instruction = ' '.join(instruction)

                embedding = generate_embeddings(model, instruction)
                new_data.append([file.split('.csv')[0], instruction, embedding])

        new_data_df = pd.DataFrame(new_data, columns=['Sample Name', 'Instruction', 'Embedding'])
        new_data_df.to_parquet('./data/winwebsec.parquet')

if __name__=='__main__':
    main(sys.argv[1])

    