# 

This repository contains the code accompanying the paper "_Days of Future Past_: Towards Robust Detection of Malware Variants via LLM-based Embedding Generation".

It provides implementations to re-execute the models described in the paper. 
To run the code, you need PyTorch >= 2

### Repository Structure
The repository is organized as follows:
* config/ – experimental configurations,
* losses/ – implementations of the loss functions,
* models/ – source code for the models,
* trainers/ – trainer class for training and testing models,
* utility/ – utility functions

### Running Experiments
**main.py** contains the code to run experiments. 
To execute an experiment, pass the configuration file as an argument:

'''
python/python3 main.py config/config.json
'''

The main.py script imports the required libraries and sets environment variables to enable GPU usage.

### Configuration Parameters

All experiment parameters must be defined in the configuration file. Key parameters include:

* **generate_csv**: =1 to generate CSV files from assembly instructions,
* **generate_embeddings**: =1 to generate embeddings of the assembly instructions by using a language model,
* **section_to_analyze**: section of the malware file to analyze,
* **language_model_id**: identifier of the language model for extracting latent representations,
* **data_area_folder**: folder containing the data,
* **benign_data_folder**: folder containing benign samples,
* **malware_data_folder**: folder containing malware samples

#### Training and Testing Options
* **train_initial_detector**: =1 to train the base detector,
* **test_initial_detector**: =1 to test the base detector,
*  **train_vae**: =1 to train the VAE (neural data augmentation),
*  **test_vae**: =1 to test the VAE,
*  **train_detector**: =1 to train the detector with the augmented dataset,
*  **generate_variants**: =1 to generate variants using the VAE,
*  **do_oversampling_vae**: =1 to train the detector with both oversampled and VAE-generated samples,
*  **do_oversampling**: =1 to train the detector with only oversampled data,
*  **n_oversampling**: number of oversampled samples to generate

#### Dataset and Experiment Setting
*  **family_to_consider**: target malware family,
*  **n_mal_samples_train**: number of malware samples in the original training set,
*  **n_ben_samples_train**: number of benign samples in the original training set,
*  **random_sampling_malware**: =0 for chronological split of the training and test sets,
*  **seed**: random seed for reproducibility

#### Training Hyperparameters

*  **batch_size**: batch size,
*  **n_gpu**: number of the GPU to use,
*  **lr_detector**: learning rate for the detector,
*  **lr_vae**: learning rate for the VAE,
*  **epochs_detector**: number of epochs for training the detector,
*  **epochs_vae**: number of epochs for training the VAE,
*  **n_variants_to_generate**: number of VAE variants to generate,

For the experiments, we collected samples of malicious software from the MOTIF dataset (https://github.com/boozallen/MOTIF).