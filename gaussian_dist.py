# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:39:42 2019

@author: dell
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

X = np.random.normal(0,1,(1200,2))
#X = dataset[0]

pt = PowerTransformer()
X = pt.fit(X).transform(X)

fig = plt.figure()


outliers_fraction = 0.1

Y = np.zeros((60, 20))
Z = np.zeros((60, 20))
Z_pred = np.zeros((60, 20))
Z_pred_3sigma = np.zeros((60, 20))
evr = np.zeros((2, 20))
Thershold = np.zeros(20)


#ax = fig.add_subplot(1, 2, 1)
#ax.cla()
#
#ax.scatter(X[:, 0], X[:, 1], color='black', s=2)
#ax.set_xlim(-3.5,3.5)
#ax.set_ylim(-3.5,3.5)
  
ax = fig.add_subplot(1, 1, 1)
ax.cla()

z_max = []
z_min = []
z_mean = []
z_std = []

for i in range(2, 60):
    
    algorithm = LocalOutlierFactor(n_neighbors=i, contamination = 0.1)
    y = algorithm.fit_predict(X)
    
    z = -algorithm.negative_outlier_factor_
        
    z_max.append(np.max(z))
    z_min.append(np.min(z))
    z_mean.append(np.mean(z))
    z_std.append(np.std(z))
    
x=list(range(2,60))      
ax.plot(x, z_max, color='black', linestyle='dashed')
ax.plot(x, z_min, color='black', linestyle='dotted')
ax.errorbar(x, z_mean, yerr=z_std, color='black', linestyle='solid')

ax.legend(('max', 'min', 'mean with std'),
               loc='upper right')    

 
#ax.set_title('PCA outlier %s, outliers_fraction = %.2f' % (str(i+1),outliers_fraction))
ax.set_ylabel('value of LOF', fontsize=12)
ax.set_xlabel('k',fontsize=12)
ax.set_xlim(0,60)
ax.set_ylim(0,5)
    

plt.show()