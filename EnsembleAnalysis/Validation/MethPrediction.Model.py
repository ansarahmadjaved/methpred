import pandas as pd
from scipy import stats
from scipy.spatial import distance
# import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import pickle


from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, mean_squared_error

import os
import argparse

# 15736

parser = argparse.ArgumentParser(description='Process CpG sites.')
parser.add_argument('--start', type=int, required=True, help='Starting CpG index')
parser.add_argument('--end', type=int, required=True, help='Ending CpG index')
args = parser.parse_args()

fromCpgs = args.start
toCpgs = args.end

print(fromCpgs,'-', toCpgs)

TPMscaler = StandardScaler()

# ### GTEX based DATA
print("Reading Methylation Data")
TrainMeth = pd.read_pickle('../GtexBasedPrediction/GtexMeth.common.pkl').T
print("Reading Tpm Data")
TrainTpm = pd.read_pickle('../GtexBasedPrediction/GtexTPM.common.pkl').T
        
# #### TCGA BASED DATA
# print("Reading Tpm Data")
# TrainMeth = pd.read_pickle('../GenesBasedPrediction/TrainMeth.pkl')
# print("Reading Methylation Data")
# TrainTpm = pd.read_pickle('../GenesBasedPrediction/TrainTpm.pkl')


with open('../GenesBasedPrediction/CpgsGeneDict.pkl', 'rb') as f:
    CpgsGeneDict = pickle.load(f)
print("Starting Models")

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

models = []

np.random.seed(42)
# Linear Models
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))

# # Tree-based Models
models.append(('DecisionTree', DecisionTreeRegressor()))
models.append(('RandomForest ', RandomForestRegressor(n_estimators=2, n_jobs=10)))
models.append(('GradientBoosting', GradientBoostingRegressor()))

# Support Vector Regressor
models.append(('SVR', SVR()))

# ### TCGA
# TopCpgs = list(pd.read_csv('TopCPGs.csv')['CPG'])
# TopCsvDf = pd.read_pickle('TopCsvDf.pkl')

# #### GTEX based
TopCsvDf = pd.read_pickle('../GtexBasedPrediction/GtexCsvDf.pkl')
TopCsvDf = TopCsvDf[TopCsvDf['PearsonR'] > 0.5]
TopCpgs = TopCsvDf.index
# print('Error is in Reading CSV files')

Dataset = pd.read_pickle('ValidationCohorts/Immunotherapy/RaviLC.mRNA.pkl')
Dataset = Dataset
# print('Error is in DataFrame')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
filename = "RaviLC.GTEX"
try: os.mkdir(filename)
except:pass
i = fromCpgs
models = dict(models)
ResultList = []
i = 0
for Cpg in TopCpgs[fromCpgs:toCpgs]:
    Genes = CpgsGeneDict[Cpg]
    # print('Error is with CPG format')
    Model = TopCsvDf.loc[Cpg]['Model']
    model = models[Model]
    try:
        model.fit(TPMscaler.fit_transform(TrainTpm[Genes].fillna(0)), TrainMeth[Cpg].fillna(0))  
        predictedMeth = model.predict(TPMscaler.fit_transform(Dataset[Genes].fillna(0))) 
        MethylDict = {}
        MethylDict['Cpg'] = Cpg
        k = 0
        for m in Dataset.index:
            MethylDict[m] = predictedMeth[k]
            k += 1
        ResultList.append(MethylDict)
        df = pd.DataFrame(ResultList)
        df = df.T
        df.columns = df.loc['Cpg']
        df = df.iloc[1:]
        df.to_csv(f'{filename}/{filename}.{fromCpgs}.to.{toCpgs}.csv', index=True)

        if i % 10 == 0:    
            print(i)
        i += 1    
        
    # ResultList.append(MethylDict)        
    except:pass
