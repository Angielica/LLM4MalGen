import os
import json
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import datetime

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

    epochs_detector = params['epochs_detector']
    do_oversampling_vae = params['do_oversampling_vae']
    do_oversampling = params['do_oversampling']
    n_oversampling = params['n_oversampling']
    n_mal_samples_train = params['n_mal_samples_train']
    n_ben_samples_train = params['n_ben_samples_train']
    family_to_consider = params['family_to_consider']
    results_list = []

    _current_date = datetime.datetime.now()

    exp = f'epochs_detector_{epochs_detector}_n_mal_samples_{n_mal_samples_train}_ben_samples_{n_ben_samples_train}_family_{family_to_consider}_oversampling_{do_oversampling}_oversampling_vae_{do_oversampling_vae}_num_over_samples_{n_oversampling}'
    params['log_path'] = f'./results/log_{exp}.txt'

    print('[INFO] Load dataset.')
    with open(params['log_path'], 'a') as f:
        f.write(f'\n[INFO] Set of experiments --> Date: {_current_date}\n')
        f.write('\n [INFO] Load dataset.')

    benign_data_path = os.path.join(data_area_folder, 'benign.parquet')
    benign_df = pd.read_parquet(benign_data_path)

    print(f'There are {benign_df.shape[0]} benign samples')

    malware_data_path = os.path.join(data_area_folder, 'malware.parquet')
    malware_df = pd.read_parquet(malware_data_path)
    malware_df = malware_df[malware_df['Family'] == family_to_consider]

    print(f'\n There are {malware_df.shape[0]} malware samples belonging to the family: {family_to_consider}')
    train_malware_df = malware_df.sample(n=n_mal_samples_train, random_state=seed)
    test_malware_df = malware_df.drop(train_malware_df.index)

    train_benign_df = benign_df.sample(n=n_ben_samples_train, random_state=seed)
    test_benign_df = benign_df.drop(train_benign_df.index)

    train = pd.concat([train_malware_df, train_benign_df])
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)

    test = pd.concat([test_malware_df, test_benign_df])
    test = test.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f'\n TRAIN: There are {train_benign_df.shape[0]} benign samples and {train_malware_df.shape[0]} malware samples')
    print(f'\n TEST: There are {test_benign_df.shape[0]} benign samples and {test_malware_df.shape[0]} malware samples')

    with open(params['log_path'], 'a') as f:
        f.write(f'\n There are {benign_df.shape[0]} benign samples. \n')
        f.write(f'\n There are {malware_df.shape[0]} malware samples belonging to the family: {family_to_consider}\n')
        f.write(f'\n TRAIN: There are {train_benign_df.shape[0]} benign samples and {train_malware_df.shape[0]} malware samples\n')
        f.write(f'\n TEST: There are {test_benign_df.shape[0]} benign samples and {test_malware_df.shape[0]} malware samples \n')
        f.write('\n [INFO] Training initial detector.\n')

    train_dataset = EmbeddingDataset(train)
    test_dataset = EmbeddingDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    params['in_channels'] = train['Embedding'].values[0].shape[0]
    params['hidden_channels'] = [512, 256]

    params['last_detector_checkpoint_path'] = f'./models/initial_detector_{exp}.pt'
    params['training_loss_path'] = f'./plots/training_initial_detector_loss_{exp}.png'

    detector = Detector(params['in_channels'], params['hidden_channels']).to(device)
    trainer = TrainerDetector(params, detector)

    threshold = params['threshold']

    if params['train_initial_detector']:
        print('[INFO] Training initial detector.')
        trainer.train(train_loader)

    if params['test_initial_detector']:
        print('[INFO] Testing initial detector.')
        with open(params['log_path'], 'a') as f:
            f.write('\n [INFO] Testing initial detector. \n')

        detector.load_state_dict(torch.load(params['last_detector_checkpoint_path'] ))

        y_score, y_true = trainer.test(test_loader, detector)
        y_score, y_true = y_score.numpy(), y_true.numpy()

        _auc = roc_auc_score(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall, precision)

        print(f'\n AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} ')
        with open(params['log_path'], 'a') as f:
            f.write(f'\n AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} \n')

        y_pred = [1 if score >= threshold else 0 for score in y_score]

        report = classification_report(y_true, y_pred)
        print(report)
        with open(params['log_path'], 'a') as f:
            f.write(report)

        report = classification_report(y_true, y_pred, output_dict=True)

        precision_0 = report['0.0']['precision']
        recall_0 = report['0.0']['recall']
        f1_score_0 = report['0.0']['f1-score']
        support_0 = report['0.0']['support']

        precision_1 = report['1.0']['precision']
        recall_1 = report['1.0']['recall']
        f1_score_1 = report['1.0']['f1-score']
        support_1 = report['1.0']['support']

        accuracy = report['accuracy']
        macro_avg_precision = report['macro avg']['precision']
        macro_avg_recall = report['macro avg']['recall']
        macro_avg_f1_score = report['macro avg']['f1-score']
        macro_avg_support = report['macro avg']['support']

        weighted_avg_precision = report['weighted avg']['precision']
        weighted_avg_recall = report['weighted avg']['recall']
        weighted_avg_f1_score = report['weighted avg']['f1-score']
        weighted_avg_support = report['weighted avg']['support']


        results_list.append(['Init detector', epochs_detector, precision_0, recall_0, f1_score_0, support_0,
                             precision_1, recall_1, f1_score_1, support_1, accuracy,
                             macro_avg_precision, macro_avg_recall, macro_avg_f1_score, macro_avg_support,
                             weighted_avg_precision, weighted_avg_recall, weighted_avg_f1_score, weighted_avg_support,
                             auc_pr, _auc])

    if params['do_oversampling']:
        print('Train Detector with only oversampling.')

        train_malware_df_aug = train_malware_df.copy()
        for i in range(n_oversampling):
            train_malware_df_aug = pd.concat([train_malware_df_aug, train_malware_df.copy()])

        train_malware_df_aug = train_malware_df_aug.reset_index(drop=True)

        train_aug = pd.concat([train_malware_df, train_malware_df_aug, train_benign_df])
        train_aug = train_aug.sample(frac=1, random_state=seed).reset_index(drop=True)
        print('\n [Detector with only oversampling] Oversampling variants: ', train_malware_df_aug.shape[0])

        with open(params['log_path'], 'a') as f:
            f.write('\n[INFO] Train and Test detector with oversampling enabled.\n')
            f.write(f'\n[Detector with only oversampling] Oversampling variants: {train_malware_df_aug.shape[0]}\n')

        train_aug_dataset = EmbeddingDataset(train_aug)
        train_aug_loader = DataLoader(train_aug_dataset, batch_size=params['batch_size'], shuffle=True)

        params['last_detector_checkpoint_path'] = f'./models/detector_with_only_oversampling_{exp}.pt'
        params['training_loss_path'] = f'./plots/training_detector_with_only_oversampling_loss_{exp}.png'

        detector = Detector(params['in_channels'], params['hidden_channels']).to(device)
        trainer = TrainerDetector(params, detector)

        trainer.train(train_aug_loader)

        detector.load_state_dict(torch.load(params['last_detector_checkpoint_path']))

        _y_score, y_true = trainer.test(test_loader, detector)
        _y_score, y_true = _y_score.numpy(), y_true.numpy()

        _auc = roc_auc_score(y_true, _y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, _y_score)
        auc_pr = auc(recall, precision)

        print(f'AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} ')
        with open(params['log_path'], 'a') as f:
            f.write(f'\n AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} \n')

        y_pred = [1 if score >= threshold else 0 for score in _y_score]

        report = classification_report(y_true, y_pred)
        print(report)
        with open(params['log_path'], 'a') as f:
            f.write(report)

        report = classification_report(y_true, y_pred, output_dict=True)

        precision_0 = report['0.0']['precision']
        recall_0 = report['0.0']['recall']
        f1_score_0 = report['0.0']['f1-score']
        support_0 = report['0.0']['support']

        precision_1 = report['1.0']['precision']
        recall_1 = report['1.0']['recall']
        f1_score_1 = report['1.0']['f1-score']
        support_1 = report['1.0']['support']

        accuracy = report['accuracy']
        macro_avg_precision = report['macro avg']['precision']
        macro_avg_recall = report['macro avg']['recall']
        macro_avg_f1_score = report['macro avg']['f1-score']
        macro_avg_support = report['macro avg']['support']

        weighted_avg_precision = report['weighted avg']['precision']
        weighted_avg_recall = report['weighted avg']['recall']
        weighted_avg_f1_score = report['weighted avg']['f1-score']
        weighted_avg_support = report['weighted avg']['support']

        results_list.append(['Detector with oversampling', epochs_detector, precision_0, recall_0, f1_score_0, support_0,
                             precision_1, recall_1, f1_score_1, support_1, accuracy,
                             macro_avg_precision, macro_avg_recall, macro_avg_f1_score, macro_avg_support,
                             weighted_avg_precision, weighted_avg_recall, weighted_avg_f1_score, weighted_avg_support,
                             auc_pr, _auc])

    vae = VAE(p_dims=[2,256,params['in_channels']]).to(device)
    trainer_vae = TrainerVAE(params, vae)

    train_malware_dataset = EmbeddingDataset(train_malware_df)
    train_malware_loader = DataLoader(train_malware_dataset, shuffle=True)

    params['last_vae_checkpoint_path'] = f'./models/vae_{exp}.pt'
    params['training_loss_path'] = f'./plots/training_vae_loss_{exp}.png'

    if params['train_vae']:
        with open(params['log_path'], 'a') as f:
            f.write('\n[INFO] Train VAE.\n')
        trainer_vae.train(train_malware_loader)

    if params['generate_variants']:
        vae.load_state_dict(torch.load(params['last_vae_checkpoint_path']))

        n_variants_to_generate = params['n_variants_to_generate']

        variants, z_variants = [], []

        for n in range(n_variants_to_generate):
            for x, _ in train_malware_loader:
                v, z_v = vae.generate_mu_sigma(x)
                variants.append(v.numpy()[0])
                z_variants.append(z_v.numpy()[0])

        variants = pd.DataFrame({'Embedding': [row.tolist() for row in variants]})
        variants['Label'] = 1

        if params['do_oversampling_vae']:
            n_oversampling = params['n_oversampling']
            train_malware_df_aug = train_malware_df.copy()

            for i in range(n_oversampling):
                train_malware_df_aug = pd.concat([train_malware_df_aug, train_malware_df.copy()])
            train_malware_df_aug = train_malware_df_aug.reset_index(drop=True)
            print('Oversampling variants: ', train_malware_df_aug.shape[0])
            train = pd.concat([train_malware_df, train_malware_df_aug, train_benign_df, variants])

            with open(params['log_path'], 'a') as f:
                f.write('\n[INFO] Train and Test detector after adding VAE-generated variants and oversampling.\n')
                f.write(f'\n[INFO] Oversampling variants: {train_malware_df_aug.shape[0]} \n')
                f.write(f'\n[INFO] VAE-generated variants: {variants.shape[0]}\n')

            train = train.sample(frac=1, random_state=seed).reset_index(drop=True)

            train_dataset = EmbeddingDataset(train)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            params['last_detector_checkpoint_path'] = f'./models/detector_with_vae_and_oversampling_{exp}.pt'
            params['training_loss_path'] = f'./plots/training_detector_with_vae_and_oversampling_loss_{exp}.png'

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
            with open(params['log_path'], 'a') as f:
                f.write(f'\n AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} \n')

            y_pred = [1 if score >= threshold else 0 for score in y_score]

            report = classification_report(y_true, y_pred)
            print(report)
            with open(params['log_path'], 'a') as f:
                f.write(report)

            report = classification_report(y_true, y_pred, output_dict=True)

            precision_0 = report['0.0']['precision']
            recall_0 = report['0.0']['recall']
            f1_score_0 = report['0.0']['f1-score']
            support_0 = report['0.0']['support']

            precision_1 = report['1.0']['precision']
            recall_1 = report['1.0']['recall']
            f1_score_1 = report['1.0']['f1-score']
            support_1 = report['1.0']['support']

            accuracy = report['accuracy']
            macro_avg_precision = report['macro avg']['precision']
            macro_avg_recall = report['macro avg']['recall']
            macro_avg_f1_score = report['macro avg']['f1-score']
            macro_avg_support = report['macro avg']['support']

            weighted_avg_precision = report['weighted avg']['precision']
            weighted_avg_recall = report['weighted avg']['recall']
            weighted_avg_f1_score = report['weighted avg']['f1-score']
            weighted_avg_support = report['weighted avg']['support']

            results_list.append(['Detector with VAE and Oversampling', epochs_detector, precision_0, recall_0, f1_score_0, support_0,
                                 precision_1, recall_1, f1_score_1, support_1, accuracy,
                                 macro_avg_precision, macro_avg_recall, macro_avg_f1_score, macro_avg_support,
                                 weighted_avg_precision, weighted_avg_recall, weighted_avg_f1_score,
                                 weighted_avg_support, auc_pr, _auc])


        train = pd.concat([train_malware_df, train_benign_df, variants])
        with open(params['log_path'], 'a') as f:
            f.write('\n[INFO] Train and Test detector after adding VAE-generated variants.\n')
            f.write(f'\n[INFO] VAE-generated variants: {variants.shape[0]}\n')

        train = train.sample(frac=1, random_state=seed).reset_index(drop=True)

        train_dataset = EmbeddingDataset(train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        params['last_detector_checkpoint_path'] = f'./models/detector_with_vae_{exp}.pt'
        params['training_loss_path'] = f'./plots/training_detector_with_vae_loss_{exp}.png'

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
        with open(params['log_path'], 'a') as f:
            f.write(f'\n AUC-PR: {auc_pr:.4f}, AUC: {_auc:.4f} \n')

        y_pred = [1 if score >= threshold else 0 for score in y_score]

        report = classification_report(y_true, y_pred)
        print(report)
        with open(params['log_path'], 'a') as f:
            f.write(report)

        report = classification_report(y_true, y_pred, output_dict=True)

        precision_0 = report['0.0']['precision']
        recall_0 = report['0.0']['recall']
        f1_score_0 = report['0.0']['f1-score']
        support_0 = report['0.0']['support']

        precision_1 = report['1.0']['precision']
        recall_1 = report['1.0']['recall']
        f1_score_1 = report['1.0']['f1-score']
        support_1 = report['1.0']['support']

        accuracy = report['accuracy']
        macro_avg_precision = report['macro avg']['precision']
        macro_avg_recall = report['macro avg']['recall']
        macro_avg_f1_score = report['macro avg']['f1-score']
        macro_avg_support = report['macro avg']['support']

        weighted_avg_precision = report['weighted avg']['precision']
        weighted_avg_recall = report['weighted avg']['recall']
        weighted_avg_f1_score = report['weighted avg']['f1-score']
        weighted_avg_support = report['weighted avg']['support']

        results_list.append(['Detector with VAE', epochs_detector, precision_0, recall_0, f1_score_0, support_0,
                             precision_1, recall_1, f1_score_1, support_1, accuracy,
                             macro_avg_precision, macro_avg_recall, macro_avg_f1_score, macro_avg_support,
                             weighted_avg_precision, weighted_avg_recall, weighted_avg_f1_score, weighted_avg_support,
                             auc_pr, _auc])

    results_df = pd.DataFrame(results_list, columns=['Model', 'Epochs Detector', 'Precision-0', 'Recall-0', 'F1-Score-0',
                                                     'Support-0', 'Precision-1', 'Recall-1', 'F1-Score-1', 'Support-1',
                                                     'Accuracy', 'Macro-avg-precision', 'Macro-avg-recall', 'Macro-avg-f1-score',
                                                     'Macro-avg-support', 'Weighted-avg-precision', 'Weighted-avg-recall',
                                                     'Weighted-avg-f1-score', 'Weighted-avg-support', 'AUPRC', 'AUC'])
    results_df.to_csv(f'./results/results_{exp}.csv', index=False)


if __name__=='__main__':
    main(sys.argv[1])

    
