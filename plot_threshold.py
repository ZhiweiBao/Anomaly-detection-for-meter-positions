# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:30:33 2019

@author: dell
"""
import numpy as np
from outliers import smirnov_grubbs as grubbs
batch_no = '261806084535853'


fig = plt.figure()
for i in range(20):
    ax = fig.add_subplot(4,5,i+1)
    x = np.array(range(1,61))

    z = Z[:,i]
    z0 = z-1
    zz = np.append(z0,-z0)+1

#    
    mu = np.mean(zz)
    sigma = np.std(zz, ddof=1)
#    
    thershold = mu+3*sigma

    Thershold[i] = thershold
    z_pred_3sigma = np.int64(z<thershold)
    Z_pred_3sigma[:, i] = z_pred_3sigma


    y=thershold + x*0
    ax.scatter(x,z, color='black', s=10)
    ax.plot(x,y,color='black',linestyle='dashed')
    ax.set_title('BTACH_NO = %s' % batch_no)
    ax.set_ylabel('value of LOF')
    ax.set_xlabel('Position')
    #plt.set_xlim(-8,8)
    ax.set_ylim(0.8,8)
#ax.set_ylim(0.8,3.5)
plt.show()



e_p_LOF = np.zeros((20,3))
Z_sort=np.sort(Z.reshape(1200))

for i in range(20):
    e_p_LOF[i,0] = np.where(Z==Z_sort[-i-1])[1]+1
    e_p_LOF[i,1] = np.where(Z==Z_sort[-i-1])[0]+1
    e_p_LOF[i,2] = Z_sort[-i-1]

G = []

for i in range(20):
    
    z = Z[:,i]
    z0 = z-1
    zz = np.append(z0,-z0)+1
#    g = grubbs.test(zz, alpha=0.05)
    g = grubbs.max_test_outliers(zz, alpha=0.05)
    G.append(g)
    
    
    x = np.mean(zz)
    S = np.std(zz, ddof=1)
    g_value = (g[0] - x)/S
    
    zz = np.delete(np.sort(zz), -1)



