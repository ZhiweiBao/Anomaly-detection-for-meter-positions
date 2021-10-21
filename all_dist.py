import cx_Oracle
import pandas as pd
import numpy as np

import pre_process

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab 
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

ora_conn = cx_Oracle.connect('SDDX1_BASE', '123456', 'localhost:1521/orcl')
df_ora = pd.read_sql('select distinct '
                     'B.DETECT_EQUIP_NO,'
                     'B.POSITION_NO,'
                     'B.BAR_CODE,'
                     'B.LOAD_CURRENT,'
                     'B.PF,'
                     'B.AVE_ERR '
                     'from MT_BASICERR_MET_CONC B, MT_MODEL M '
                     'where B.DETECT_TASK_NO=M.DETECT_TASK_NO '
                     'and B.AVE_ERR < 0.6 and B.AVE_ERR >-0.6 '                   
                     'and M.ARRIVE_BATCH_NO=261806084535853', con=ora_conn)
ora_conn.close()


dataset = pre_process.get_raw_number(df_ora)

fig = plt.figure()


X = dataset[13]
for i in range(60):
    
    #-----------------------------------------------------------------------------
    ax = fig.add_subplot(111)
    if i not in [13,14,27,40,53,59]:
             

    #----normal_data-----------------------------------------------        
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[i],cumulative=True,color='black')
#        ax.set_title('%s' %str(i+1))
    #    ax.set_ylabel('%')
    #    ax.set_xlabel('X')
#        ax.set_xlim(-0.3,0.5)
#        ax.set_ylim(0,10)
    #--------------------------------------------------- 
    
    elif i == 13:
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[14],cumulative=True,color='r')
        
    elif i == 14:
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[13],cumulative=True,color='r')
    
    elif i == 27:
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[53],cumulative=True,color='r')
        
    elif i == 53:
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[27],cumulative=True,color='black')
    
    elif i == 40:
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[59],cumulative=True,color='r')
        
    elif i == 59:
#        ax = fig.add_subplot(6, 10, i+1)
#        ax.cla()     
    #    sns.distplot(X[i],bins=30,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "black" })
        sns.kdeplot(X[40],cumulative=True,color='black')
    
    
#    result = np.zeros((60,2))
#    for k in range(60):
#        re = stats.ks_2samp(normal_data, X[k])
#        result[k][0] = re[0]
#        result[k][1] = re[1]
#    ax.set_ylim(-8,8)
    
    # plt.savefig("figure/outlier/%s.png" % str(i+1))


#Z_diff=Z-Z_qualified
#Z_divide = Z/Z_qualified
    
plt.show()


