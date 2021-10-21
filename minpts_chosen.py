import cx_Oracle
import pandas as pd
import numpy as np

import pre_process

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
df_ora = pd.read_sql('select '
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


dataset = pre_process.data_reform(df_ora)

fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')
#colors = cm.rainbow(np.arange(20) / 20)

#xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
#                     np.linspace(-7, 7, 150))

outliers_fraction = 0.1
#Y_IF = np.zeros((60, 20))
#Z_IF = np.zeros((60, 20))
Y = np.zeros((60, 20))
Z = np.zeros((60, 20))
Z_pred = np.zeros((60, 20))

for j in range(20):

    ax = fig.add_subplot(4, 5, j+1)
    ax.cla()
    X = dataset[j]
    if np.isnan(X.min()):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X)
        X = imp.transform(X)
    
    pt = PowerTransformer()
    X = pt.fit(X).transform(X)
    
#    qt = QuantileTransformer(output_distribution='normal')
#    X = qt.fit_transform(X)
    
#    X = np.random.normal(0,1,(100,60))
    z_max = []
    z_min = []
    z_mean = []
    z_std = []
    
    
    for i in range(2, 61):
    
        algorithm = LocalOutlierFactor(n_neighbors=i, contamination = 0.1)
        y = algorithm.fit_predict(X)
    
        z = -algorithm.negative_outlier_factor_
        
        z_max.append(np.max(z))
        z_min.append(np.min(z))
        z_mean.append(np.mean(z))
        z_std.append(np.std(z))
    
    x=list(range(2,61))    
    ax.plot(x, z_max, color='black', linestyle='dashed')
    ax.plot(x, z_min, color='black', linestyle='dotted')
    ax.errorbar(x, z_mean, yerr=z_std, color='black', linestyle='solid')

    ax.legend(('max', 'min', 'mean with std'),
               loc='upper right') 
    
    
    
    ax.set_title('%s' % str(j+1))
    #ax.set_ylabel('Y')
    #ax.set_xlabel('X')
    #ax.set_xlim(-8,8)
    ax.set_ylim(0.8,2)
#plt.savefig("figure/outlier/%s.png" % str(i+1))


plt.show()


