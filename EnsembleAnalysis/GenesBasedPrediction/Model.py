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

parser = argparse.ArgumentParser(description='Process CpG sites.')
parser.add_argument('--start', type=int, required=True, help='Starting CpG index')
parser.add_argument('--end', type=int, required=True, help='Ending CpG index')
args = parser.parse_args()

fromCpgs = args.start
toCpgs = args.end

print(fromCpgs,'-', toCpgs)

TPMscaler = StandardScaler()
Methscaler = StandardScaler()

print("Reading Tpm Data")
TrainMeth = pd.read_pickle('TrainMeth.pkl')
ValMeth = pd.read_pickle('ValMeth.pkl')

print("Reading Methylation Data")
TrainTpm = pd.read_pickle('TrainTpm.pkl')
ValTpm = pd.read_pickle('ValTpm.pkl')

        
with open('CpgsGeneDict.pkl', 'rb') as f:
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


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
ResultList = []
try: os.mkdir('Results')
except:pass
i = fromCpgs
for SelectedCpg in list(CpgsGeneDict.items())[fromCpgs:toCpgs]:
    for model in models:        
        Model = model[1]
        Cpg = SelectedCpg[0]
        Genes = SelectedCpg[1]
        Model.fit(TPMscaler.fit_transform(TrainTpm[Genes].fillna(0)), TrainMeth[Cpg].fillna(0))  
        predictedMeth = Model.predict(TPMscaler.transform(ValTpm[Genes].fillna(0))) 
        resultDict = {'CPG':Cpg, "PearsonR":stats.pearsonr(predictedMeth,ValMeth[Cpg].fillna(0))[0], 
        "Euclidean Distance" : distance.euclidean(predictedMeth,ValMeth[Cpg].fillna(0)), 
        'MSE': mean_squared_error(predictedMeth,ValMeth[Cpg].fillna(0)),
        "MAE": mean_absolute_error(predictedMeth,ValMeth[Cpg].fillna(0)),
        "R2" : r2_score(predictedMeth,ValMeth[Cpg].fillna(0)), 
        'P.Val' : stats.pearsonr(predictedMeth,ValMeth[Cpg].fillna(0))[1], 
        'Model':model[0], "Genes":Genes, 
        }
        ResultList.append(resultDict)
        pd.DataFrame(ResultList).to_csv(f'Results/EnsembleModel.{fromCpgs}.to.{toCpgs}.csv')
    if i % 100 == 0:    
        print(i ,"Out of " ,toCpgs)
    i += 1    

        
