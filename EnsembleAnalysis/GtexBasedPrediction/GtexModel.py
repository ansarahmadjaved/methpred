import pandas as pd
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, mean_squared_error

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import pickle

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

print("Reading Files")

with open('../GenesBasedPrediction/CpgsGeneDict.pkl', 'rb') as f:
    CpgsGeneDict = pickle.load(f)


GtexMeth = pd.read_pickle('GtexMeth.common.pkl')
GtexMeth = GtexMeth.T
GtexTpm = pd.read_pickle('GtexTPM.common.pkl')
GtexTpm = GtexTpm.T
GtexTpm = GtexTpm.groupby(GtexTpm.columns, axis=1).mean()

GtexCommonCpgs = list(GtexMeth.columns)

print("Starting Models")


ResultList = []
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
ResultList = []
try: os.mkdir('Results.GTEX.prediction')
except:pass

X = GtexTpm.reset_index(drop=True).fillna(0)
y = GtexMeth.reset_index(drop=True).fillna(0)
TrainTPM, TestTPM , TrainMeth, TestMeth = train_test_split(X, y, train_size=0.5, random_state=42, )
i = fromCpgs
for SelectedCpg in list(CpgsGeneDict.items())[fromCpgs:toCpgs]:
    for model in models:        
        Model = model[1]
        try:
            Cpg = SelectedCpg[0]
            Genes = SelectedCpg[1]
            if Cpg in GtexCommonCpgs:
                Model.fit(TPMscaler.fit_transform(TrainTPM[Genes]) ,TrainMeth[Cpg])
                predictedMeth = Model.predict(TPMscaler.transform(TestTPM[Genes]))
                
                resultDict = {'CPG':Cpg, "PearsonR":stats.pearsonr(predictedMeth,TestMeth[Cpg])[0], 
                "Euclidean Distance" : distance.euclidean(predictedMeth,TestMeth[Cpg]), 
                'MSE': mean_squared_error(predictedMeth,TestMeth[Cpg]),
                "MAE": mean_absolute_error(predictedMeth,TestMeth[Cpg]),
                "R2" : r2_score(predictedMeth,TestMeth[Cpg]), 
                'P.Val' : stats.pearsonr(predictedMeth,TestMeth[Cpg])[1], 
                'Model':model[0], "Genes":Genes, 
                "CPG.Mean":GtexMeth[Cpg].mean(), "CPG.Var":GtexMeth[Cpg].var(),
                "No.Genes":len(Genes)}
                ResultList.append(resultDict)
            else:
                pass
        except:pass
        pd.DataFrame(ResultList).to_csv(f'Results.GTEX.prediction/GTEX.{fromCpgs}.to.{toCpgs}.csv')
    if i % 100 == 0:    
        print(i ,"Out of " ,toCpgs)
        i += 1    