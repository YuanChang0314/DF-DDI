import os
import torch
import pandas as pd
import numpy as np
import rdkit
import rdkit.Geometry
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import argparse
from torch_geometric.data import DataLoader
from tqdm import tqdm
from model import GNN_graphpred
from sklearn.metrics import roc_auc_score
from deepforest import CascadeForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score,accuracy_score,auc,precision_recall_curve
from tdc.multi_pred import DDI
from imblearn.over_sampling import SMOTE
import random

data = DDI(name = "drugbank")
split = data.get_split()
train = split['train']
test = split['test']
print('tdc done')
lentr = len(train)
lente = len(test)



i = 0
while i < lentr:
    random_drug1 = random.choice(train['Drug1_ID'])
    random_drug2 = random.choice(train['Drug2_ID'])
    while ((train['Drug1_ID'] == random_drug1) & (train['Drug2_ID'] == random_drug2)).any():
        random_drug1 = random.choice(train['Drug1_ID'])
        random_drug2 = random.choice(train['Drug2_ID'])
    
    new_row = [
        random_drug1,
        train.loc[train['Drug1_ID'] == random_drug1, 'Drug1'].values[0],
        random_drug2,
        train.loc[train['Drug2_ID'] == random_drug2, 'Drug2'].values[0],
        0
    ]
    
    train = pd.concat([train, pd.DataFrame([new_row], columns=train.columns)], ignore_index=True)
    i += 1



i = 0
while i < lente:
    random_drug1 = random.choice(test['Drug1_ID'])
    random_drug2 = random.choice(test['Drug2_ID'])
    while ((test['Drug1_ID'] == random_drug1) & (test['Drug2_ID'] == random_drug2)).any():
        random_drug1 = random.choice(test['Drug1_ID'])
        random_drug2 = random.choice(test['Drug2_ID'])
    
    new_row = [
        random_drug1,
        test.loc[test['Drug1_ID'] == random_drug1, 'Drug1'].values[0],
        random_drug2,
        test.loc[test['Drug2_ID'] == random_drug2, 'Drug2'].values[0],
        0
    ]
    
    test = pd.concat([test, pd.DataFrame([new_row], columns=test.columns)], ignore_index=True)
    i += 1


train.to_csv('./data/trainforstage1.csv')
test.to_csv('./data/testforstage1.csv')
