from sklearn import preprocessing
import pandas as pd

class preprocess:
    
    def drop_features(dataset):
        print('--Hapus Atribut dimulai--')
        #hapus fitur yang tidak digunakan
        #fitur dihapus setelah ada proses chi square, prosesnya terpisah dan tidak dalam program utama
        #karena waktu proses yang lama
        features_drop = ['nameOrig','newbalanceOrig','oldbalanceDest','newbalanceDest'] 
        df = dataset.drop(columns = features_drop)
        print('--Hapus Atribut Selesai--')
        return df
    
    def features_encode(dataset):
        #encoding categorical features
        print('')
        print('--Encode atribut dimulai--')
        lbenc = preprocessing.LabelEncoder()
        dataset['nameDest']= lbenc.fit_transform(dataset['nameDest'])
        dataset['type']= lbenc.fit_transform(dataset['type'])
        print('--Encode atribut selesai--')
        return dataset
    
    def random_undersampling(dataset):
        print('--Random Under-Sampling dimulai--')
        countClass0, countClass1 = dataset.isFraud.value_counts()
        dfClass0 = dataset[dataset['isFraud'] == 0]
        dfClass1 = dataset[dataset['isFraud'] == 1]
        dfClass_0_Under = dfClass0.sample(countClass1)
        data = pd.concat([dfClass_0_Under, dfClass1], axis=0)
        print('--Random Under-Sampling selesai--')
        return data