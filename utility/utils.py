import re
import pandas as pd

import torch


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
            parts = rest.split('\t')
            bytecode = ''
            instruction = ''
            operands = ''

            if len(parts) == 1:
                bytecode = parts[0].strip()
            elif len(parts) >= 2:
                bytecode = parts[0].strip()
                instr_parts = parts[1].strip().split(None, 1)  # Split on first space
                if instr_parts:
                    instruction = instr_parts[0]
                    if len(instr_parts) > 1:
                        operands = instr_parts[1]

            row = [f'.{current_section}', address, bytecode, instruction, operands]
            rows.append(row)

    df = pd.DataFrame(rows, columns=['Section', 'Address', 'Bytecode', 'Instruction', 'Operands'])
    df.to_csv(df_file_path, index=False)


def generate_embeddings(model, sentences):
    with torch.no_grad():
        embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings
