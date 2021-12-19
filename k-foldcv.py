# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:29:29 2021

@author: user
"""

import pandas as pd
import pickle
import time
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv(r'E:\BISMILLAH TUGAS AKHIR\kodingan TA\Bismillah Kodingan TA ABI\FIX\Data\money_transaction.csv')

features_drop = ['nameOrig','newbalanceOrig','oldbalanceDest','newbalanceDest'] 
df = df.drop(columns = features_drop)
df.head()

lbenc = preprocessing.LabelEncoder()
df['nameDest']= lbenc.fit_transform(df['nameDest'])
df['type']= lbenc.fit_transform(df['type'])

countClass0, countClass1 = df.isFraud.value_counts()
dfClass0 = df[df['isFraud'] == 0]
dfClass1 = df[df['isFraud'] == 1]
dfClass_0_Under = dfClass0.sample(countClass1)
data = pd.concat([dfClass_0_Under, dfClass1], axis=0)
X = df.drop('isFraud', axis=1)
X = X.dropna()
y = df.isFraud
#X dan y menggunakan variabel data ketika ingin menguji dengan Random Under-sampling


kf = KFold(n_splits=10, random_state=42, shuffle=True)
i = 1
acc_temp = 0
sensi_temp = 0
speci_temp = 0

# model = pickle.load(open('E:\BISMILLAH TUGAS AKHIR\kodingan TA\Bismillah Kodingan TA ABI\FIX\gbt-model.sav', 'rb'))
for train_index, test_index in kf.split(X):
    start_time = time.time()
    print("Train :", len(train_index), ", Test : ", len(test_index))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    gb_clf = GradientBoostingClassifier(n_estimators = 100)
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    akurasi = round(((tp + tn) / y_pred.__len__()) * 100, 2)
    sensi = round(tp / (tn + fn) * 100, 2)
    speci = round(tn / (tn + fp) * 100, 2)
    execution_time = round((time.time() - start_time),2)
    print('k :',i," = ",(tn, fp, fn, tp), ', acc = ',akurasi, ', sensi =', sensi, '. speci = ', speci, ', time = ', execution_time)
    
    i+=1
    # temp_tp += tp
    acc_temp += akurasi
    sensi_temp += sensi
    speci_temp += speci

print(acc_temp)
print("Akurasi :" , acc_temp/10)
print("Sensitivitas :" , sensi_temp/10)
print("Spesifisitas :" , speci_temp/10)