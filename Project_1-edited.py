import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.plotting import parallel_coordinates

missing_values = ["n/a", "na", "--", "#N/A"]
df = pd.read_excel('DATA_PERFORMANCE_Human.xlsx', na_values = missing_values)
df.head()

print (df['absen_semangat'].isnull())

print (df.isnull().sum())
df.dropna(inplace =True)



to_drop = ['Nama orang', 'Pangkat', 'talent_cluster', 'indeks_kesehatan']
df.drop(to_drop, inplace=True, axis=1)

print (df.isnull().sum())
print("DF: \n", df)

X = df.drop(['performansi_individu', 'engagement_berjuang','absen_tertekan','absen_nyaman','absen_semangat'], axis = 1)
X.values.astype(float)
X.head()
print("X: ", X)
y = df['performansi_individu']
print("Y: \n",y)
y.values.astype(float)
y.head()

regressor = SVR(kernel = 'linear')
regressor.fit(X.values, y.values)

yPred = regressor.predict(X.values)

print ("Score :", regressor.score(X.values, yPred))
print ("MSE : %.2f" % mean_squared_error(y.values, yPred))

print(X.values)

# VARIABLE TEST
nilai_kompetensi = 2
nilai_behavior = 1.7
engagement_ucapan = 3
engagement_tinggal= 2.2


Xnew = [[nilai_kompetensi, nilai_behavior, engagement_ucapan, engagement_tinggal]]
ynew = regressor.predict(Xnew)

print("Predict ",ynew)





