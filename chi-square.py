import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
from google.colab import drive
drive.mount("/content/gdrive")

class chiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-value
        self.chi2 = None #Chi Test Statistic
        self.dof = None 
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
        print(result)
    
    def TestIndependence(self,colX,colY, alpha=0.5):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        print('p-value :', self.p)
        print('chi2 :', self.chi2)
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

df = pd.read_csv('/content/gdrive/My Drive/money_transaction.csv')
#initialize chi square class
cT = chiSquare(df)

testColumns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'nameDest', 'newbalanceDest', 'isFlaggedFraud']
for var in testColumns :
    cT.TestIndependence(colX = var, colY = "isFraud")
