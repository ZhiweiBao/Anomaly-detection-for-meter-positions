# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:56:11 2019

@author: dell
"""
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

fig = plt.figure()
quantiles=[]
interval=[]
Z_pred_3sigma = np.zeros((60, 20))
Z_pred_quan = np.zeros((60, 20))
Thershold = np.zeros(20)

for i in range(20):
    ax = fig.add_subplot(4, 5, i+1)
    z = Z[:,i]
    z0 = z-1
    zz = np.append(z0,-z0)+1
    ax.hist(zz,60)
#    
    mu = np.mean(zz)
    sigma = np.std(zz)
#    
    thershold = mu+3*sigma
    Thershold[i] = thershold
    z_pred_3sigma = np.int64(z<thershold)
    Z_pred_3sigma[:, i] = z_pred_3sigma
    
    
plt.show()



for i in range(20):
    z = Z[:,i]
#    z0 = z-1
#    zz = np.append(z0,-z0)+1
    quantiles.append(stats.mstats.mquantiles(z))
    q1 = quantiles[i][0]
    q3 = quantiles[i][2]
    delta_q = q3-q1
    low_interval = q1-3*delta_q
    up_interval = q3+3*delta_q
    interval.append([low_interval, up_interval, low_interval+up_interval])
    
    z_pred_quan = np.int64(z>low_interval) * np.int64(z<up_interval)
    Z_pred_quan[:, i] = z_pred_quan



fig = plt.figure()
ax = fig.add_subplot(111)
z = Z[:,0]
z0 = z-1
zz = np.append(z0,-z0)+1
ax.hist(zz,60,color='black')
ax.set_ylabel('Pt(%)')
ax.set_xlabel('value of LOF')
plt.show()
