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



parser = argparse.ArgumentParser(description='About deepforest ddi')
parser.add_argument('--dataset', type=str, default='drugbank',help='The dataset that the model is going to use for training the model')
parser.add_argument('--prediction_type', type=int, default=49, help='Which type of interaction is going to be the positive type')
parser.add_argument('--model', type=str, default= None, help='The deep forest model that used for predicition')
parser.add_argument('--outputfile', type=str, default= None, help='The file that the model is going to be saved at')
parser.add_argument('--input_drug1', type=str, default= None, help='The first drug that used for predicting the interaction')
parser.add_argument('--input_drug2', type=str, default= None, help='The Second drug that used for predicting the interaction')
parser.add_argument('--n_estimators', type=int, default= 2, help='Specify the number of estimators in each cascade layer')
parser.add_argument('--n_trees', type=int, default= 100, help='Specify the number of trees in each estimator')
parser.add_argument('--max_layers', type=int, default= 20, help='Specify the maximum number of cascade layers.')
parser.add_argument('--use_predictor', type=bool, default= False, help='Using the predictor or not, should be True is predictor is not None')
parser.add_argument('--predictor', type=str, default= "forest", help='Specify the type of the predictor, should be one of "forest", "xgboost", "lightgbm"')
parser.add_argument('--n_tolerant_rounds', type=int, default= 2, help='Specify the number of tolerant rounds when handling early stopping, should be larger or equal to 1')
args = parser.parse_args()



# Load drug interaction data from TDC
data = DDI(name = args.dataset)
split = data.get_split()
train = split['train']
test = split['test']
print('tdc done')
train = pd.read_csv('data/newstage1/new1train.csv',index_col=0)
test = pd.read_csv('data/newstage1/new1test.csv',index_col=0)
#i = 0
#while i < 80000:
#    random_drug1 = random.choice(train['Drug1_ID'])
#    random_drug2 = random.choice(train['Drug2_ID'])
#    while ((train['Drug1_ID'] == random_drug1) & (train['Drug2_ID'] == random_drug2)).any():
#        random_drug1 = random.choice(train['Drug1_ID'])
#        random_drug2 = random.choice(train['Drug2_ID'])
#    
#    new_row = [
#        random_drug1,
#        train.loc[train['Drug1_ID'] == random_drug1, 'Drug1'].values[0],
#        random_drug2,
#        train.loc[train['Drug2_ID'] == random_drug2, 'Drug2'].values[0],
#        0
#    ]
#    
#    train = pd.concat([train, pd.DataFrame([new_row], columns=train.columns)], ignore_index=True)
#    i += 1
#
#i = 0
#while i < 10000:
#    random_drug1 = random.choice(test['Drug1_ID'])
#    random_drug2 = random.choice(test['Drug2_ID'])
#    while ((test['Drug1_ID'] == random_drug1) & (test['Drug2_ID'] == random_drug2)).any():
#        random_drug1 = random.choice(test['Drug1_ID'])
#        random_drug2 = random.choice(test['Drug2_ID'])
#    
#    new_row = [
#        random_drug1,
#        test.loc[test['Drug1_ID'] == random_drug1, 'Drug1'].values[0],
#        random_drug2,
#        test.loc[test['Drug2_ID'] == random_drug2, 'Drug2'].values[0],
#        0
#    ]
#    
#    test = pd.concat([test, pd.DataFrame([new_row], columns=test.columns)], ignore_index=True)
#    i += 1

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
        attention_scores1 /= torch.sqrt(torch.tensor(hidden_dim).float())
        weights1 = torch.softmax(attention_scores1, dim=1)
        weighted_sum1 = torch.matmul(weights1, V2)
        
        # Attention from drug2 to drug1
        Q2 = self.Q_layer(drug2)
        K1 = self.KV_layer(drug1)
        V1 = self.KV_layer(drug1)
        attention_scores2 = torch.matmul(Q2, K1.t())
        attention_scores2 /= torch.sqrt(torch.tensor(hidden_dim).float())
        weights2 = torch.softmax(attention_scores2, dim=1)
        weighted_sum2 = torch.matmul(weights2, V1)
        
        return weighted_sum1, weighted_sum2 #Both with shape of [hidden_dim,]








# Training Data processing 
x = np.array(train["Drug1_ID"])
x1 = list(x)
y = np.array(train["Drug2_ID"])
y1 = list(y)
z = np.append(x,y)
ftrain = np.unique(z)
ftrain = list(ftrain)
removelist = []
alldrugtrain = {}
for i in ftrain:
    try:
        index1 = x1.index(i)
        drug1 = train.iloc[index1,0]
        try:
            a = train.iloc[index1,1]
            drug_fingerprints = gnnsmile(a)#getting the numerical representation of the drug smiles sequence, dimensionality for drug vector is 300
            drug_fingerprints = list(drug_fingerprints[0])
            alldrugtrain[drug1] = drug_fingerprints
        except:
            removelist.append(i)
    except:
        index2 = y1.index(i) 
        drug2 = train.iloc[index2,2]
        try:
            a = train.iloc[index2,3]
            drug_fingerprints = gnnsmile(a)
            drug_fingerprints = list(drug_fingerprints[0])
            alldrugtrain[drug2] = drug_fingerprints
        except:
            removelist.append(i)
for x in removelist:
    ftrain.remove(x)





# Testing Data processing 
x = np.array(test["Drug1_ID"])
x1 = list(x)
y = np.array(test["Drug2_ID"])
y1 = list(y)
z = np.append(x,y)
ftest= np.unique(z)
ftest = list(ftest)
removelist = []
alldrugtest = {}
for i in ftest:
    try:
        index1 = x1.index(i)
        drug1 = test.iloc[index1,0]
        try:
            a = test.iloc[index1,1]
            drug_fingerprints = gnnsmile(a)
            drug_fingerprints = list(drug_fingerprints[0])
            alldrugtest[drug1] = drug_fingerprints           
        except:
            removelist.append(i)
    except:
        index2 = y1.index(i) 
        drug2 = test.iloc[index2,2]
        try:
            a = test.iloc[index2,3]
            drug_fingerprints = gnnsmile(a)
            drug_fingerprints = list(drug_fingerprints[0])
            alldrugtest[drug2] = drug_fingerprints
        except:
            removelist.append(i)
for x in removelist:
    ftest.remove(x)
keylisttrain = list(alldrugtrain.keys())
keylisttest = list(alldrugtest.keys())
X_train = []
Y_train = []


print('dp done')

# Attention to drug vector pair
device = torch.device("cuda:" + "0") if torch.cuda.is_available() else torch.device("cpu")
for i in range(len(train)):
    try:
        input_dim = 300 #Dimension of the drug matrix getting by the GNN and Bert
        hidden_dim = 50 #Dimension of attention score that create by the cross attention model    
        cross_attention_model = CrossAttention(input_dim, hidden_dim)
        cross_attention_model.to(device)
        drug1_tensor = torch.tensor(alldrugtrain[train.iloc[i,0]], dtype=torch.float32,device='cuda:0').unsqueeze(0) 
        drug2_tensor = torch.tensor(alldrugtrain[train.iloc[i,2]], dtype=torch.float32,device='cuda:0').unsqueeze(0)
        result_attention1, result_attention2 = cross_attention_model(drug1_tensor, drug2_tensor)
        result_attention1 = result_attention1.cpu()#getting the attention result of drug1 & drug2
        result_attention2 = result_attention2.cpu()
        X_train.append(alldrugtrain[train.iloc[i,0]]+alldrugtrain[train.iloc[i,2]]+result_attention1.tolist()[0] + result_attention2.tolist()[0])#concatenated the attention score to the original data
        #Final Vector Dimension is 700
        if train.iloc[i,4] >= 1:# getting the label for positive or negative samples
            Y_train.append(1)
        else:
            Y_train.append(0)
    except:
        print(1)
        continue
X_test = []
Y_test = []
for i in range(len(test)):
    try:
        input_dim = 300
        hidden_dim = 50        
        cross_attention_model = CrossAttention(input_dim, hidden_dim)
        cross_attention_model.to(device)
        drug1_tensor = torch.tensor(alldrugtest[test.iloc[i,0]], dtype=torch.float32,device='cuda:0').unsqueeze(0)
        drug2_tensor = torch.tensor(alldrugtest[test.iloc[i,2]], dtype=torch.float32,device='cuda:0').unsqueeze(0)
        result_attention1, result_attention2 = cross_attention_model(drug1_tensor, drug2_tensor)
        result_attention1 = result_attention1.cpu()
        result_attention2 = result_attention2.cpu()
        X_test.append(alldrugtest[test.iloc[i,0]]+alldrugtest[test.iloc[i,2]]+result_attention1.tolist()[0] + result_attention2.tolist()[0])
        if test.iloc[i,4] >= 1:
            Y_test.append(1)
        else:
            Y_test.append(0)
    except:
        continue

print('at done')



smo = SMOTE()
x_resample,y_resample = smo.fit_resample(X_train,Y_train)

smo = SMOTE()
X_test,Y_test = smo.fit_resample(X_test,Y_test)




# Training the Classifier
model = CascadeForestClassifier(n_estimators = args.n_estimators,n_trees = args.n_trees,max_layers = args.max_layers,
                                use_predictor = args.use_predictor, predictor = args.predictor,
                                n_tolerant_rounds = args.n_tolerant_rounds)
print(len(x_resample),len(y_resample))
if args.model != None:
    model.load(args.model)
else:
    model.fit(x_resample,y_resample )
    print('train done')


# loading the Model
#if args.outputfile != None:
#model.save('savemodelstage1')   


# evaluates the trained classifier on the testing data and prints various performance
# metrics such as AUC, accuracy, precision, recall, and F1 score.
Y_pred = model.predict(X_test)
Y_score = model.predict_proba(X_test)

acc = np.array(confusion_matrix(Y_test, Y_pred))
precision, recall, thresholds = precision_recall_curve(Y_test, Y_score[:,1])
print('tp is',acc[1][1])
print('tn is',acc[0][0])
print('accuracy is',accuracy_score(Y_test,Y_pred))
print('percision is', acc[1][1]/(acc[1][1]+acc[0][1]))
print('recall is', acc[1][1]/(acc[1][1]+acc[1][0]))
print('f1score is',f1_score(Y_test,Y_pred)) 
aucscore = roc_auc_score(Y_test,Y_score[:,1])
print('aucscore is',aucscore)
print("aupr is",auc(recall, precision))

# Result of testing
if args.input_drug1 and args.input_drug2 != None:
    drug_fingerprints1 = gnnsmile(args.input_drug1)
    drug_fingerprints2 = gnnsmile(args.input_drug2)    
    cross_attention_model = CrossAttention(300, 50)
    cross_attention_model.to(device)
    drug1_tensor = torch.tensor(alldrugtrain[drug_fingerprints1], dtype=torch.float32,device='cuda:0').unsqueeze(0) 
    drug2_tensor = torch.tensor(alldrugtrain[drug_fingerprints2], dtype=torch.float32,device='cuda:0').unsqueeze(0)
    result_attention1, result_attention2 = cross_attention_model(drug1_tensor, drug2_tensor)
    result_attention1 = result_attention1.cpu()
    result_attention2 = result_attention2.cpu()
    testingvector = drug_fingerprints1 + drug_fingerprints2 + result_attention1 + result_attention2
    testingresult = model.predict(testingvector)
    print('the interaction of',args.input_drug1,'and',args.input_drug2,'is:',testingresult)
