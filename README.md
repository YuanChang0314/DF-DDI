
## Installation
We used the following Python packages for core development. We tested on `Python 3.11`.
```
pytorch                   1.0.1
torch-cluster             1.2.4              
torch-geometric           1.0.3
torch-scatter             1.1.2 
torch-sparse              0.2.4
torch-spline-conv         1.0.6
rdkit                     2019.03.1.0
tqdm                      4.31.1
tensorboardx              1.6
```

## Dataset download
All the necessary data files can be downloaded from the following links.

For the drug smiles dataset, download from PyTDC: https://tdcommons.ai/start/

## Tokenizer Training
```
python vqvae.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting tokenizer to `OUTPUT_MODEL_PATH`.

## Pre-training and fine-tuning
#### 1. Self-supervised pre-training
```
python pretrain.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 2. Fine-tuning
```
python finetune.py --model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`





