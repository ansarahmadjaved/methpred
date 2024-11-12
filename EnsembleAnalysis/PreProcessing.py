import pandas as pd
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, mean_squared_error

# fromCpgs = input('Starting CPGs :')
# toCpgs = input('Ending CPGs :')

# Different Scalers for transcripts and methylations

os.chdir('..')

print('Reading Files')

MethTCGA = pd.read_pickle('/home/sbl/Documents/TCGA/Pan Cancer/pancan_methylation.pkl')
MethTCGA = MethTCGA.T


TCGAtpm = pd.read_pickle('TCGA.Kallisto.TPM.pkl')
TCGAtpm = TCGAtpm.set_index('sample')
CommPatients = MethTCGA.columns.intersection(TCGAtpm.columns)
MethTCGA = MethTCGA[CommPatients]
TCGAtpm = TCGAtpm[CommPatients]

TCGAtpm = TCGAtpm.astype('float32')
MethTCGA = MethTCGA.astype('float32')

TCGAtpm = (2 ** TCGAtpm) - 0.001

MethTCGA = MethTCGA.T
TCGAtpm = TCGAtpm.T

# Methylation Annotations
Meth450Annotations = pd.read_table('HM450.hg38.manifest.gencode.v22.tsv.gz', compression='gzip', low_memory=False, sep='\t')
Meth450Annotations = Meth450Annotations.dropna(subset='transcriptIDs')
Meth450Annotations = Meth450Annotations.drop_duplicates(subset='probeID', keep='first')
Meth450Annotations = Meth450Annotations.set_index('probeID')


# Selecting Common CPGS in TCGA and Annotation files
CommonCPGs = list(Meth450Annotations.index.intersection(MethTCGA.columns))
Meth450Annotations = Meth450Annotations.loc[CommonCPGs]

# Selecting Protein Coding Genes
ProteinCodCpgsAnnotations = Meth450Annotations[Meth450Annotations['transcriptTypes'].str.contains('protein_coding', case=False)]

ProteinCodCpgsAnnotations.loc[:, 'transcriptIDs'] = ProteinCodCpgsAnnotations['transcriptIDs'].str.split(';')
ProteinCodCpgsAnnotations = ProteinCodCpgsAnnotations.explode('transcriptIDs')
CommTranscripts = TCGAtpm.columns[TCGAtpm.columns.isin(ProteinCodCpgsAnnotations['transcriptIDs'])]
ProteinCodCpgsAnnotations = ProteinCodCpgsAnnotations[ProteinCodCpgsAnnotations['transcriptIDs'].isin(CommTranscripts)]

CpgsGeneDict = ProteinCodCpgsAnnotations['transcriptIDs'].to_dict()
print(len(CpgsGeneDict))


#### Splitting Based on Cancer Subtype
TCGAphenotype = pd.read_table('TCGA_phenotype_denseDataOnlyDownload.tsv.gz', compression='gzip')
TCGAphenotype = TCGAphenotype.set_index('sample')
TCGAphenotype = TCGAphenotype.loc[CommPatients]
TrainTpmIdx, TempTpmIdx , TrainMethIdx, TempMethIdx = train_test_split(TCGAphenotype.index, TCGAphenotype.index, train_size=0.5, random_state=42, stratify=TCGAphenotype['_primary_disease'])
ValTpmIdx, TestTpmIdx , ValMethIdx, TestMethIdx =  train_test_split(TempTpmIdx, TempMethIdx, train_size=0.5, random_state=42, stratify=TCGAphenotype.loc[TempTpmIdx]['_primary_disease'])


TrainTpm = TCGAtpm.loc[TrainTpmIdx]
ValTpm = TCGAtpm.loc[ValTpmIdx]
TestTPM = TCGAtpm.loc[TestTpmIdx]


TrainMeth = MethTCGA.loc[TrainMethIdx]
ValMeth = MethTCGA.loc[ValMethIdx]
TestMeth = MethTCGA.loc[TestMethIdx]


TrainMeth = TrainMeth.to_pickle('TrainMeth.pkl')
ValMeth = ValMeth.to_pickle('ValMeth.pkl')
TestMeth = TestMeth.to_pickle('TestMeth.pkl')

TrainTpm = TrainTpm.to_pickle('TrainTpm.pkl')
ValTpm = ValTpm.to_pickle('ValTpm.pkl')
TestTPM = TestTPM.to_pickle('TestTPM.pkl')


ProteinCodCpgsAnnotations.to_csv('ProteinCodCpgsAnnotations.csv')
with open('CpgsGeneDict.pkl', 'wb') as f:
    pickle.dump(CpgsGeneDict, f)
