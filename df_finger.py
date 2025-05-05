import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc.multi_pred import DDI
from deepforest import CascadeForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score,accuracy_score,auc,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
data = DDI(name = 'drugbank')
split = data.get_split()

#data1 = DDI(name = 'TWOSIDES')
#split1 = data1.get_split()

print(split.keys())
train = split['train']
test = split['test']
valid = split['valid']


time0 = time.time()

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
            targetmol1 = Chem.MolFromSmiles(a)
            fps1 = AllChem.GetMorganFingerprintAsBitVect(targetmol1,2,nBits = 1024)
            fingerprints1 = fps1.ToBitString()
            drug_fingerprints = []
            for k in range(1024):
                drug_fingerprints.append(float(fingerprints1[k]))
            alldrugtrain[drug1] = drug_fingerprints
        except:
            removelist.append(i)
    except:
        index2 = y1.index(i) 
        drug2 = train.iloc[index2,2]
        try:
            a = train.iloc[index2,3]
            targetmol1 = Chem.MolFromSmiles(a)
            fps1 = AllChem.GetMorganFingerprintAsBitVect(targetmol1,2,nBits = 1024)
            fingerprints1 = fps1.ToBitString()
            drug_fingerprints = []
            for k in range(1024):
                drug_fingerprints.append(float(fingerprints1[k]))
            alldrugtrain[drug2] = drug_fingerprints
        except:
            removelist.append(i)
for x in removelist:
    ftrain.remove(x)





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
            targetmol1 = Chem.MolFromSmiles(a)
            fps1 = AllChem.GetMorganFingerprintAsBitVect(targetmol1,2,nBits = 1024)
            fingerprints1 = fps1.ToBitString()
            drug_fingerprints = []
            for k in range(1024):
                drug_fingerprints.append(float(fingerprints1[k]))
            alldrugtest[drug1] = drug_fingerprints
        except:
            removelist.append(i)
    except:
        index2 = y1.index(i) 
        drug2 = test.iloc[index2,2]
        try:
            a = test.iloc[index2,3]
            targetmol1 = Chem.MolFromSmiles(a)
            fps1 = AllChem.GetMorganFingerprintAsBitVect(targetmol1,2,nBits = 1024)
            fingerprints1 = fps1.ToBitString()
            drug_fingerprints = []
            for k in range(1024):
                drug_fingerprints.append(float(fingerprints1[k]))
            alldrugtest[drug2] = drug_fingerprints
        except:
            removelist.append(i)
for x in removelist:
    ftest.remove(x)




keylisttrain = list(alldrugtrain.keys())
keylisttest = list(alldrugtest.keys())


X_train = []
Y_train = []
tr1 = 0
tr0 = 0
te1 = 0
te0 = 0
for i in range(len(train)):
    try:
        X_train.append(alldrugtrain[train.iloc[i,0]]+alldrugtrain[train.iloc[i,2]])
        if train.iloc[i,4] == 73:
            Y_train.append(1)
            tr1 += 1
        else:
            Y_train.append(0)
            tr0 += 1
    except:
        continue
print("train finished")
time1 = time.time()
print(time1-time0)
print(len(X_train[0]))
print(len(X_train[0]))
print(len(X_train[0]))
X_test = []
Y_test = []
for i in range(len(test)):
    try:
        X_test.append(alldrugtest[test.iloc[i,0]]+alldrugtest[test.iloc[i,2]])
        if test.iloc[i,4] == 73:
            Y_test.append(1)
            te1 += 1
        else:
            Y_test.append(0)
            te0 += 1
    except:
        continue
print("test finished")
time1 = time.time()
print(time1-time0)



model = CascadeForestClassifier()
model.fit(X_train,Y_train)
print("model done")
time1 = time.time()
print(time1-time0)


Y_pred = model.predict(X_test)
Y_score = model.predict_proba(X_test)
aucscore = roc_auc_score(Y_test,Y_score[:,1])
print('aucscore is',aucscore)
acc = np.array(confusion_matrix(Y_test, Y_pred))
precision, recall, thresholds = precision_recall_curve(Y_test, Y_score[:,1])
print("aupr is",auc(recall, precision))
print('tp is',acc[1][1])
print('tn is',acc[0][0])
print('accuracy is',accuracy_score(Y_test,Y_pred))
print('percision is', acc[1][1]/(acc[1][1]+acc[0][1]))
print('recall is', acc[1][1]/(acc[1][1]+acc[1][0]))
print('f1score is',f1_score(Y_test,Y_pred))