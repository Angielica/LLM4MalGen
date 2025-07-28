import os
import json
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

from models.detector import Detector
from models.vae import VAE
from trainers.trainer import *
from utility.utils import *
from utility.dataloader import *


def main(fname):
    with open(fname) as f:
        params = json.load(f)

    seed = params['seed']
    set_reproducibility(seed)

    n_gpu = params['n_gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = n_gpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    data_area_folder = params['data_area_folder']
    benign_data_folder = params['benign_data_folder']
    malware_data_folder = params['malware_data_folder']

    if not os.path.exists(os.path.join(data_area_folder, 'CSV')):
        os.mkdir(os.path.join(data_area_folder, 'CSV'))
        os.mkdir(os.path.join(data_area_folder, 'CSV', benign_data_folder))
        os.mkdir(os.path.join(data_area_folder, 'CSV', malware_data_folder))

    dir = ['./models', './plots', './results']
    for d in dir:
        if not os.path.exists(d):
            os.mkdir(d)

    # Convert exe file into csv file
    if params['generate_csv']:

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

    if params['generate_embeddings']:
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

    benign_data_path = os.path.join(data_area_folder, 'benign.parquet')
    benign_df = pd.read_parquet(benign_data_path)

    print(f'There are {benign_df.shape[0]} benign samples')

    family_to_consider = params['family_to_consider']

    malware_data_path = os.path.join(data_area_folder, 'malware.parquet')
    malware_df = pd.read_parquet(malware_data_path)
    malware_df = malware_df[malware_df['Family'] == family_to_consider]

    print(f'There are {malware_df.shape[0]} malware samples belonging to the family: {family_to_consider}')

    train_malware_df = malware_df.sample(n=params['n_mal_samples_train'], random_state=seed)
    test_malware_df = malware_df.drop(train_malware_df.index)

    train_benign_df = benign_df.sample(n=params['n_ben_samples_train'], random_state=seed)
    test_benign_df = benign_df.drop(train_benign_df.index)

    train = pd.concat([train_malware_df, train_benign_df])
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)

    test = pd.concat([test_malware_df, test_benign_df])
    test = test.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f'TRAIN: There are {train_benign_df.shape[0]} benign samples and {train_malware_df.shape[0]} malware samples')
    print(f'TEST: There are {test_benign_df.shape[0]} benign samples and {test_malware_df.shape[0]} malware samples')

    train_dataset = EmbeddingDataset(train)
    test_dataset = EmbeddingDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    params['in_channels'] = train['Embedding'].values[0].shape[0]
    params['hidden_channels'] = [512, 256]

    params['last_initial_detector_checkpoint_path'] = './models/initial_detector.pt'
    params['training_loss_path'] = './plots/training_initial_detector_loss.png'
    params['log_path'] = './results/log.txt'

    detector = Detector(params['in_channels'], params['hidden_channels']).to(device)
    trainer = TrainerDetector(params, detector)

    threshold = params['threshold']

    if params['train_initial_detector']:
        open(params['log_path'], 'w').close()
        trainer.train(train_loader)

    if params['test_initial_detector']:
        detector.load_state_dict(torch.load(params['last_initial_detector_checkpoint_path'] ))

        y_score, y_true = trainer.test(test_loader, detector)
        y_score, y_true = y_score.numpy(), y_true.numpy()

        _auc = roc_auc_score(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall, precision)

        print(f'AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} ')
        with open(params['log_path'], 'w') as f:
            f.write(f'AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} \n')

        y_pred = [1 if score >= threshold else 0 for score in y_score]

        report = classification_report(y_true, y_pred)
        print(report)
        with open(params['log_path'], 'w') as f:
            f.write(report)

    vae = VAE(p_dims=[2,256,params['in_channels']]).to(device)
    trainer_vae = TrainerVAE(params, vae)

    train_malware_dataset = EmbeddingDataset(train_malware_df)
    train_malware_loader = DataLoader(train_malware_dataset, shuffle=True)

    params['last_vae_checkpoint_path'] = './models/vae.pt'
    params['training_loss_path'] = './plots/training_vae_loss.png'

    if params['train_vae']:
        trainer_vae.train(train_malware_loader)

    if params['generate_variants']:
        vae.load_state_dict(torch.load(params['last_vae_checkpoint_path']))
        n_variants_to_generate = params['n_variants_to_generate']
        variants = vae.generate(n_variants_to_generate)
        variants = variants.numpy()
        variants = pd.DataFrame({'Embedding': [row.tolist() for row in variants]})
        variants['Label'] = 1

        train = pd.concat([train_malware_df, train_benign_df, variants])
        train = train.sample(frac=1, random_state=seed).reset_index(drop=True)

        train_dataset = EmbeddingDataset(train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        params['last_detector_checkpoint_path'] = './models/detector.pt'
        params['training_loss_path'] = './plots/training_detector_loss.png'

        detector = Detector(params['in_channels'], params['hidden_channels']).to(device)
        trainer = TrainerDetector(params, detector)

        if params['train_detector']:
            trainer.train(train_loader)

        detector.load_state_dict(torch.load(params['last_detector_checkpoint_path']))

        y_score, y_true = trainer.test(test_loader, detector)
        y_score, y_true = y_score.numpy(), y_true.numpy()

        _auc = roc_auc_score(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall, precision)

        print(f'AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} ')
        with open(params['log_path'], 'w') as f:
            f.write(f'AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} \n')

        y_pred = [1 if score >= threshold else 0 for score in y_score]

        report = classification_report(y_true, y_pred)
        print(report)
        with open(params['log_path'], 'w') as f:
            f.write(report)


if __name__=='__main__':
    main(sys.argv[1])

    
