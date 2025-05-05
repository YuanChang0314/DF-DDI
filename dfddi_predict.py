# Import necessary libraries and modules
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
from itertools import repeat, product, chain
from torchvision.models import resnet18
import argparse
from torch_geometric.data import DataLoader
from tqdm import tqdm
from model import GNN_graphpred
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split,random_split,random_scaffold_split
import pandas as pd
from deepforest import CascadeForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score,accuracy_score,auc,precision_recall_curve
from tdc.multi_pred import DDI




parser = argparse.ArgumentParser(description='About deepforest ddi')
parser.add_argument('--model', type=str, default= None, help='The deep forest model that used for predicition')
parser.add_argument('--input_drug1', type=str, default= None, help='The first drug that used for predicting the interaction')
parser.add_argument('--input_drug2', type=str, default= None, help='The Second drug that used for predicting the interaction')
args = parser.parse_args()





# defines a dictionary of allowable features for molecule representation.
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


# converts an RDKit molecule object to a graph Data object.
def mol_to_graph_data_obj_simple(mol):
    num_atom_features = 2 
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:# mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long) # Graph connectivity in COO format with shape [2, num_edges]
        edge_attr = torch.tensor(np.array(edge_features_list),dtype=torch.long) # Edge feature matrix with shape [num_edges, num_edge_features]
    else:  # no bond
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data



# Use the pre-train Bert model to convert the molecule to vector
def convert(args, model, device, loader):
    model.eval()
    node_representations = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            _, node_representation = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        graph_representation = torch.mean(node_representation, dim=0)
        node_representations.append(graph_representation.cpu().numpy())


    return node_representations # Vector shape is [300,]




#  integrates BERT and GNN for processing SMILES strings and returns node representations
def gnnsmile(s):
    rdkit_mol = AllChem.MolFromSmiles(s)
    data = mol_to_graph_data_obj_simple(rdkit_mol)
    data.id = torch.tensor(1)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = GNN_graphpred(5, 300, 1,"last", 0.5, graph_pooling = 'mean', gnn_type = 'gin')# 5 GNN layers, 300 dimensionality of embeddings, 1 task, 0.5 dropout ratio
    model.from_pretrained('model_gin/{}.pth'.format('Mole-BERT'))
    dataset = [data]
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.to(device)
    node_representations = convert(args, model, device, data_loader) 
    return(node_representations)



# Defines a CrossAttention class, which is a neural network module for cross-attention mechanism.
class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.Q_layer = nn.Linear(input_dim, hidden_dim)
        self.KV_layer = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, drug1, drug2):
        # Attention from drug1 to drug2
        Q1 = self.Q_layer(drug1)
        K2 = self.KV_layer(drug2)
        V2 = self.KV_layer(drug2)
        attention_scores1 = torch.matmul(Q1, K2.t())
        attention_scores1 /= torch.sqrt(torch.tensor(50).float())
        weights1 = torch.softmax(attention_scores1, dim=1)
        weighted_sum1 = torch.matmul(weights1, V2)
        
        # Attention from drug2 to drug1
        Q2 = self.Q_layer(drug2)
        K1 = self.KV_layer(drug1)
        V1 = self.KV_layer(drug1)
        attention_scores2 = torch.matmul(Q2, K1.t())
        attention_scores2 /= torch.sqrt(torch.tensor(50).float())
        weights2 = torch.softmax(attention_scores2, dim=1)
        weighted_sum2 = torch.matmul(weights2, V1)
        
        return weighted_sum1, weighted_sum2 






model = CascadeForestClassifier()

if args.model != None:
    model.load(args.model)
else:
    print('error:please give model')



device = torch.device("cuda:" + "0") if torch.cuda.is_available() else torch.device("cpu")
# Result of testing
if args.input_drug1 and args.input_drug2 != None:
    drug_fingerprints1 = gnnsmile(args.input_drug1)
    drug_fingerprints1 = list(drug_fingerprints1[0])
    drug_fingerprints2 = gnnsmile(args.input_drug2)   
    drug_fingerprints2 = list(drug_fingerprints2[0]) 
    cross_attention_model = CrossAttention(300, 50)
    cross_attention_model.to(device)
    drug1_tensor = torch.tensor(drug_fingerprints1, dtype=torch.float32,device='cuda:0').unsqueeze(0)
    drug2_tensor = torch.tensor(drug_fingerprints2, dtype=torch.float32,device='cuda:0').unsqueeze(0)
    result_attention1, result_attention2 = cross_attention_model(drug1_tensor, drug2_tensor)
    result_attention1 = result_attention1.cpu()
    result_attention2 = result_attention2.cpu()
    testingvector = drug_fingerprints1 + drug_fingerprints2 + result_attention1.tolist()[0] + result_attention2.tolist()[0]
    testingvector = np.array(testingvector)
    testingresult = model.predict(testingvector.reshape(1,-1))
    print('the interaction of',args.input_drug1,'and',args.input_drug2,'is:',testingresult)
