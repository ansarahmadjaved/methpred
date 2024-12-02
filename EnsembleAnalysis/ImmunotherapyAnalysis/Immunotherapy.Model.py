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

print("Reading Methylation Data")
TrainMeth = pd.read_pickle('../GenesBasedPrediction/TrainMeth.pkl')
print("Reading Tpm Data")
TrainTpm = pd.read_pickle('../GenesBasedPrediction/TrainTpm.pkl')
        
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

TopCpgs = list(pd.read_csv('TopCPGs.csv')['CPG'])
TopCsvDf = pd.read_pickle('TopCsvDf.pkl')
HugoMelanoma = pd.read_pickle('GideMrna.pkl')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
try: os.mkdir('Results')
except:pass
i = fromCpgs
models = dict(models)
Dataset = HugoMelanoma.copy()
ResultList = []
i = 0
for Cpg in TopCpgs[fromCpgs:toCpgs]:
    Genes = CpgsGeneDict[Cpg]
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
    except:
        pass
    # ResultList.append(MethylDict)        
    df = pd.DataFrame(ResultList)
    df = df.T
    df.columns = df.loc['Cpg']
    df = df.iloc[1:]
    df.to_csv(f'Results/GideMrna.{fromCpgs}.to.{toCpgs}.csv', index=True)

    if i % 10 == 0:    
        print(i)
    i += 1    
        
