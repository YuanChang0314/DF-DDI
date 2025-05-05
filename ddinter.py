import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc.multi_pred import DDI
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
from sklearn.ensemble import RandomForestClassifier


traind = pd.read_csv(r'C:\Users\jackchang\Desktop\python_work\ddinter_downloads_code_B.csv')
traind = traind.sample(frac=1)
print(len(traind))
train = traind.iloc[:12112,:]
x = np.array(train["Drug_A"])
x1 = list(x)
y = np.array(train["Drug_B"])
y1 = list(y)
z = np.append(x,y)
ftrain = np.unique(z)
ftrain = list(ftrain)

print(len(ftrain))

drugtrans = pd.read_csv(r'C:\Users\jackchang\Desktop\python_work\drugbank_smile.csv')
transdic = {}
traindrugsmiledic = {}
for i in range(len(drugtrans)):
    transdic[drugtrans.iloc[i,1]] = drugtrans.iloc[i,2]

count = 0
nolis = []
for i in ftrain: 
    try:
        traindrugsmiledic[i] = transdic[i]
    except:
        count += 1
        nolis.append(i)
for i in nolis:
    if '(' in i:
        i = i[:i.index('(')-1]
    try:
        traindrugsmiledic[i] = transdic[i]
        count -= 1
    except:
        continue

for i in ftrain:
    try:
        index1 = x1.index(i)
        drug1 = train.iloc[index1,1]
        try:
            a = traindrugsmiledic[drug1]
            print(a)
        except:
            continue
    except:
        index2 = y1.index(i) 
        drug2 = train.iloc[index2,3]
        try:
            a = traindrugsmiledic[drug2]
            print(a)
        except:
            continue
for x in removelist:
    ftrain.remove(x)

    


