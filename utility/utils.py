import re
import pandas as pd
import numpy as np

import torch

def set_reproducibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

def generate_csv_files(params):
    data_area_folder = params['data_area_folder']
    benign_data_folder = params['benign_data_folder']
    malware_data_folder = params['malware_data_folder']

    benign_assembly = os.listdir(os.path.join(data_area_folder, benign_data_folder))
    for file in tqdm(benign_assembly, total=len(benign_assembly), desc='Benign assembly - CSV generation'):
        if 'exe.asm' in file:
            file_name = file.split('.exe.asm')[0] + '.csv'
            df_path = os.path.join(data_area_folder, 'CSV', benign_data_folder, file_name)
            exe_path = os.path.join(data_area_folder, benign_data_folder, file)
            generate_csv_for_each_sample(exe_path, df_path)

    malware_families = os.listdir(os.path.join(data_area_folder, malware_data_folder))

    for family in tqdm(malware_families, total=len(malware_families), desc='Malware families'):
        if not os.path.exists(os.path.join(data_area_folder, 'CSV', malware_data_folder, family)):
            os.mkdir(os.path.join(data_area_folder, 'CSV', malware_data_folder, family))

        mal_files = os.listdir(os.path.join(data_area_folder, malware_data_folder, family))

        for file in tqdm(mal_files, total=len(mal_files), desc=f'Family: {family} - CSV generation'):
            if '.asm' in file:
                file_name = file.split('.asm')[0] + '.csv'
                df_path = os.path.join(data_area_folder, 'CSV', malware_data_folder, family, file_name)
                exe_path = os.path.join(data_area_folder, malware_data_folder, family, file)
                generate_csv_for_each_sample(exe_path, df_path)

def generate_embeddings_file_with_llm(params):
    data_area_folder = params['data_area_folder']
    benign_data_folder = params['benign_data_folder']
    malware_data_folder = params['malware_data_folder']

    section_to_analyze = params['section_to_analyze']

    language_model_id = params['language_model_id']
    model = SentenceTransformer(language_model_id)

    # Generate embedding for benign data
    new_data = []
    benign_csv = os.listdir(os.path.join(data_area_folder, 'CSV', benign_data_folder))
    for file in tqdm(benign_csv, total=len(benign_csv), desc='Benign - Embedding generation'):
        if 'csv' in file:
            df_path = os.path.join(data_area_folder, 'CSV', benign_data_folder, file)
            df = pd.read_csv(df_path)
            df = df[df['Section'] == section_to_analyze]
            df = df.dropna(subset=['Instruction'])

            instruction = df['Instruction'].values.tolist()
            operands = df['Operands'].values.tolist()

            text_input = []
            for i, o in zip(instruction, operands):
                if not pd.isna(o):
                    text_input.append(str(i) + ' ' + str(o))
                else:
                    text_input.append(str(i))

            text_input = ' '.join(text_input)

            embedding = generate_embeddings(model, text_input)
            new_data.append([file.split('.csv')[0], text_input, embedding, 0])

    new_data_df = pd.DataFrame(new_data, columns=['Sample Name', 'Instruction', 'Embedding', 'Label'])
    new_data_df.to_parquet(os.path.join(data_area_folder, 'benign.parquet'))

    # Generate embedding for malware
    new_data = []
    malware_families = os.listdir(os.path.join(data_area_folder, 'CSV', malware_data_folder))

    for family in tqdm(malware_families, total=len(malware_families), desc='Malware families'):
        mal_files = os.listdir(os.path.join(data_area_folder, 'CSV', malware_data_folder, family))

        for file in tqdm(mal_files, total=len(mal_files), desc=f'Malware {family} - Embedding generation'):
            if 'csv' in file:
                df_path = os.path.join(data_area_folder, 'CSV', malware_data_folder, family, file)
                df = pd.read_csv(df_path)
                df = df[df['Section'] == section_to_analyze]
                df = df.dropna(subset=['Instruction'])

                instruction = df['Instruction'].values.tolist()
                operands = df['Operands'].values.tolist()

                text_input = []
                for i, o in zip(instruction, operands):
                    if not pd.isna(o):
                        text_input.append(str(i) + ' ' + str(o))
                    else:
                        text_input.append(str(i))

                text_input = ' '.join(text_input)

                embedding = generate_embeddings(model, text_input)
                new_data.append([family, file.split('.csv')[0], text_input, embedding, 1])

    new_data_df = pd.DataFrame(new_data, columns=['Family', 'Sample Name', 'Instruction', 'Embedding', 'Label'])
    new_data_df.to_parquet(os.path.join(data_area_folder, 'malware.parquet'))

def generate_csv_for_each_sample(exe_file_path, df_file_path):
    current_section = None
    rows = []

    # Match section headers like: 00411000 <.text>:
    section_header_pattern = re.compile(r'^\s*[0-9a-fA-F]+ <\.(text|data|rdata|rsrc|bss|idata|edata|reloc)>:')

    # Match lines like:  411000:  55                    push   ebp
    instruction_line_pattern = re.compile(r'^\s*([0-9a-fA-F]+):\s+(.+)$')

    with open(exe_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Check for section header
            section_match = section_header_pattern.match(line)
            if section_match:
                current_section = section_match.group(1)
                continue

            if current_section not in ('text', 'data', 'rdata', 'rsrc', 'bss', 'idata', 'edata', 'reloc'):
                continue  # Only process .text or .data sections

            # Check for instruction line
            instr_match = instruction_line_pattern.match(line)
            if not instr_match:
                continue

            address = instr_match.group(1)
            rest = instr_match.group(2)

            # Split rest into bytecode and (optional) instruction & operands
            bytecode_instr_match = re.match(r'^((?:[0-9a-fA-F]{2}(?:\s+|$))+)(.*)', rest)
            if bytecode_instr_match:
                bytecode = bytecode_instr_match.group(1).strip()
                instr_part = bytecode_instr_match.group(2).strip()

                if instr_part:
                    instr_parts = instr_part.split(None, 1)
                    instruction = instr_parts[0]
                    operands = instr_parts[1] if len(instr_parts) > 1 else ''
                else:
                    instruction = ''
                    operands = ''
            else:
                bytecode = rest.strip()
                instruction = ''
                operands = ''

            row = [f'.{current_section}', address, bytecode, instruction, operands]
            rows.append(row)

    df = pd.DataFrame(rows, columns=['Section', 'Address', 'Bytecode', 'Instruction', 'Operands'])
    df.to_csv(df_file_path, index=False)


def generate_embeddings(model, sentences):
    with torch.no_grad():
        embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings
