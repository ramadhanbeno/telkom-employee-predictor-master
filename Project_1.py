#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Mengimpor library yang diperlukan
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


# In[ ]:


# Import data ke python
missing_values = ["n/a", "na", "--", "#N/A"]
df = pd.read_excel('DATA_PERFORMANCE_Human.xlsx', na_values = missing_values)
df.head()


# In[ ]:


print (df['absen_semangat'].isnull())
print(df['absen_semangat'])


# In[ ]:


print (df.isnull().sum())


# In[ ]:


df.dropna(inplace =True)


# In[ ]:


to_drop = ['Nama orang', 'Pangkat', 'talent_cluster', 'indeks_kesehatan']
df.drop(to_drop, inplace=True, axis=1)


# In[ ]:


print (df.isnull().sum())


# In[ ]:


X = df.drop(['performansi_individu', 'engagement_berjuang'], axis = 1)
X.values.astype(float)
X.head()


# In[ ]:


y = df['performansi_individu']
y.values.astype(float)
y.head()


# In[ ]:


regressor = SVR(kernel = 'linear')
regressor.fit(X.values, y.values)

yPred = regressor.predict(X.values)

print ("Score :", regressor.score(X.values, yPred))
print ("MSE : %.2f" % mean_squared_error(y.values, yPred))


# In[ ]:


print(X.values)


# In[ ]:


Xnew = [[2, 1.7, 3, 2.2, 1, 2.1, 1.6]]
ynew = regressor.predict(Xnew)

print(ynew)


# In[ ]:


# pca = sklearnPCA(n_components=2) #2-dimensional PCA
# transformed = pd.DataFrame(pca.fit_transform(X_norm))

# plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class 1', c='red')
# plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class 2', c='blue')
# plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 3', c='lightgreen')

# plt.legend()
# plt.show()


# In[ ]:





# In[ ]:




