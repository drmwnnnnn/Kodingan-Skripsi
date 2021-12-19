import sys
sys.path.append('C:/Users/User/FIX')
from preprocess import preprocess
import pathlib
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class Model:
    
    dataset = None
    model = None
    cf = {}
    
    def __init__(self):
        print('start')
        
    def read_data(self,path):
        file = pathlib.Path(path)
        if file.exists():
            self.dataset = pd.read_csv(path)
        else:
            print('Path error : tidak ada data pada path ->',path)
        
    def get_dataset(self):
        return self.dataset

    def prepare(self):
        if self.dataset is not None:
            data = self.dataset
            data = preprocess.drop_features(data)
            data = preprocess.features_encode(data)
            self.dataset = data
            return self.dataset
        else:
            print('Dataset Kosong!')
    
    def rus(self):
        if self.dataset is not None:
            data = self.dataset
            data = preprocess.random_undersampling(data)
            self.dataset = data
            return self.dataset
            # print('di model rus')
            # print(self.dataset)
        else:
            print('rus gagal')
    
    def classify(self, dataset):
        self.dataset = dataset
        waktu = time.time()
        y = self.dataset.isFraud
        X = self.dataset.drop('isFraud', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)
        model = pickle.load(open('E:\BISMILLAH TUGAS AKHIR\kodingan TA\Bismillah Kodingan TA ABI\FIX\gbt-model.sav', 'rb'))
        y_pred = model.predict(X_test)
        #confusion matrix
        self.conf_matrix(y_test, y_pred)
        durasi = round((time.time() - waktu),2)
        print('Durasi :', durasi)

    def classifyRUS(self, dataset, jmlh_pohon):
        self.dataset = dataset
        waktu = time.time()
        X = self.dataset.drop('isFraud', axis=1)
        X = X.dropna()
        y = self.dataset.isFraud            
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)
        gb_clf = GradientBoostingClassifier(n_estimators = jmlh_pohon)
        gb_clf.fit(X_train, y_train)
        y_pred = gb_clf.predict(X_test)
        # confusion matrix
        self.conf_matrix(y_test, y_pred)
        durasi = round((time.time() - waktu),2)
        print('Durasi :', durasi)
    
    def conf_matrix(self, actual, predict):
        self.cf["TP"] = 0
        self.cf["FP"] = 0
        self.cf["TN"] = 0
        self.cf["FN"] = 0
        for x, y in zip(actual, predict):
            if y == x: #y = pred value, x = actual values
                if y == 1:
                    self.cf["TP"] += 1
                else:
                    self.cf["TN"] += 1
            else:
                if y == 1:
                    self.cf["FP"] += 1
                else:
                    self.cf["FN"] += 1
        self.akurasi = round(((self.cf["TP"] + self.cf["TN"]) / predict.__len__()) * 100, 2)
        self.sensi = round(self.cf["TP"] / (self.cf["TN"] + self.cf["FN"]) * 100, 2)
        # print(self.sensi)
        self.speci = round(self.cf["TN"] / (self.cf["TN"] + self.cf["FP"]) * 100, 2)
        # print(self.speci)
        
    def getConf_matrix(self):
        return self.cf

    def getAkurasi(self):
        return self.akurasi