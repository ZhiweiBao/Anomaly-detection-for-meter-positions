import cx_Oracle
import pandas as pd
import numpy as np
from scipy import stats

import pre_process

import matplotlib
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



for i in range(1):
    
    #-----------------------------------------------------------------------------
    X = dataset[13]
    
    normal_data = []
    p13_data = []
    p14_data = []
    p27_data = []
    p40_data = []
    
    for j in range(60):
        if j not in [13,14,53,59]:
            normal_data.extend(X[j])
        elif j == 14:
            p13_data = X[j]
        elif j == 13:
            p14_data = X[j]
        elif j == 53:
            p27_data = X[j]
        elif j == 59:
            p40_data = X[j]
             
    hist_num = 27
    #----normal_data-----------------------------------------------        
    ax = fig.add_subplot(111)
    ax.cla()     
#    sns.distplot(normal_data,bins=hist_num,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "white" })
    sns.kdeplot(normal_data,cumulative=True, color='black', label='Normal')
#    n, bins, patches = ax.hist(normal_data, hist_num, color = 'black', density=True)
#    mu = np.mean(normal_data)
#    sigma = np.std(normal_data)
#    y = mlab.normpdf(bins, mu, sigma)
#    ax.plot(bins, y)
    
#    ax.set_title('normal data')
#    ax.set_ylabel('%')
#    ax.set_xlabel('X')
#    ax.set_xlim(-0.3,0.5)
#    ax.set_ylim(0,10)
    #--------------------------------------------------- 

    #-------p13_data-------------------------------------------- 
#    ax = fig.add_subplot(221)
#    ax.cla()        
#    sns.distplot(p13_data,bins=hist_num,kde=True, kde_kws={"color":"r", "lw":3 }, hist_kws={ "color": "white" })
    sns.kdeplot(p13_data,cumulative=True, color='r', label='Postion14')

#    ax.set_title('Position14 data')
#    ax.set_ylabel('%')
#    ax.set_xlabel('X')
#    ax.set_xlim(-0.3,0.5)
#    ax.set_ylim(0,10)
    #--------------------------------------------------- 
    
    #--------p14_data------------------------------------------- 
#    ax = fig.add_subplot(111)
#    ax.cla()          
#    sns.distplot(normal_data,bins=hist_num,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "white" })
#    sns.distplot(p14_data,bins=hist_num,kde=True, kde_kws={"color":"y", "lw":3 }, hist_kws={ "color": "white" })
    
#    sns.kdeplot(normal_data,cumulative=True, color='black')
    sns.kdeplot(p14_data,cumulative=True, color='y', label='Postion15')
    
#    ax.set_title('Position15 data')
#    ax.set_ylabel('%')
#    ax.set_xlabel('X')
#    ax.set_xlim(-0.3,0.5)
#    ax.set_ylim(0,10)
    #--------------------------------------------------- 
    
    #--------p27_data------------------------------------------- 
#    ax = fig.add_subplot(111)
#    ax.cla()        
#    sns.distplot(normal_data,bins=hist_num,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "white" })
#    sns.distplot(p27_data,bins=hist_num,kde=True, kde_kws={"color":"b", "lw":3 }, hist_kws={ "color": "white" })
    
#    sns.kdeplot(normal_data,cumulative=True, color='black')
    sns.kdeplot(p27_data,cumulative=True, color='b', label='Postion28')
    
#    ax.set_title('Position28 data')
#    ax.set_ylabel('%')
#    ax.set_xlabel('X')
#    ax.set_xlim(-0.3,0.5)
#    ax.set_ylim(0,10)
    #--------------------------------------------------- 
    
    #------p40_data--------------------------------------------- 
#    ax = fig.add_subplot(111)
#    ax.cla()          
#    sns.distplot(normal_data,bins=hist_num,kde=True, kde_kws={"color":"black", "lw":3 }, hist_kws={ "color": "white" })
#    sns.distplot(p40_data,bins=hist_num,kde=True, kde_kws={"color":"g", "lw":3 }, hist_kws={ "color": "white" })
    
#    sns.kdeplot(normal_data,cumulative=True, color='black')
    sns.kdeplot(p40_data,cumulative=True, color='g', label='Postion41')
    
#    ax.set_title('Position41 data')
#    ax.set_ylabel('%')
#    ax.set_xlabel('X')
#    ax.set_xlim(-0.3,0.5)
#    ax.set_ylim(0,10)
    #--------------------------------------------------- 
    ax.set_xlim(-0.2,0.3)
    ax.set_ylim(0,1)
    legend = ax.legend()
    
#    colors = ['black','r','y','b','g']
#    legend_labels  = ['Normal', 'Postion14', 'Postion15', 'Postion28', 'Postion41']
#
#    # Create the legend patches
#    legend_patches = [matplotlib.patches.Patch(color=C, label=L, style='--') for
#                  C, L in zip(colors,legend_labels)]
#
#    # Plot the legend
#    ax.legend(handles=legend_patches)
    
#    ax.legend(('Normal', 'Postion14', 'Postion15', 'Postion28', 'Postion41'),
#               loc='upper right')   
    
    result = np.zeros((60,2))
    for k in range(60):
        re = stats.ks_2samp(normal_data, X[k])
        result[k][0] = re[0]
        result[k][1] = re[1]
    
    
plt.show()


