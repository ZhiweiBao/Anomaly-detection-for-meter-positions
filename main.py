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
Z_qualified = np.zeros((60, 20))
Z_pred_qualified = np.zeros((60, 20))

Z_pred_3sigma = np.zeros((60, 20))
#evr = np.zeros((2, 20))
Thershold = np.zeros(20)

for i in range(20):
    ax = fig.add_subplot(4, 5, i+1)
    ax.cla()
    #-----------------------------------------------------------------------------
    X = dataset[i]
    if np.isnan(X.min()):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X)
        X = imp.transform(X)

#    pt = PowerTransformer()
#    X = pt.fit(X).transform(X)

#    qt = QuantileTransformer(output_distribution='normal')
#    X = qt.fit_transform(X)
    
#    X = Normalizer().fit_transform(X)
    X = StandardScaler().fit_transform(X)
#    X = RobustScaler().fit_transform(X)
    
    pca = PCA(n_components=2, whiten=True)
    X_r = pca.fit(X).transform(X)
#    evr[:, i] = pca.explained_variance_ratio_
#    X_cov = np.cov(X)
    
    z = np.zeros((60, 40))
    for j in range(10,50):

        algorithm = LocalOutlierFactor(n_neighbors=j, contamination=outliers_fraction)
        y = algorithm.fit_predict(X)

        z[:,j-10] = -algorithm.negative_outlier_factor_  
    
    z_max = np.max(z,1)
    
    z0 = z_max-1
    zz = np.append(z0,-z0)+1

#    
    mu = np.mean(zz)
    sigma = np.std(zz)
#    
    thershold = mu+3*sigma
    Thershold[i] = thershold
    z_pred_3sigma = np.int64(z_max<thershold)
    z_pred = z_pred_3sigma
#    z_pred = np.int64(z_max<1.4)
    
    Z[:, i] = z_max
    Z_pred[:, i] = z_pred
#    #    ------------------------------------------------------
#    
#    X = dataset_qualified[i]
#    if np.isnan(X.min()):
#        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#        imp.fit(X)
#        X = imp.transform(X)
#
#    pt = PowerTransformer()
#    X = pt.fit(X).transform(X)
#
##    qt = QuantileTransformer(output_distribution='normal')
##    X = qt.fit_transform(X)
#    
##    X = Normalizer().fit_transform(X)
##    X = StandardScaler().fit_transform(X)
##    X = RobustScaler().fit_transform(X)
#    
#    pca = PCA(n_components=2, whiten=True)
#    X_r = pca.fit(X).transform(X)
##    evr[:, i] = pca.explained_variance_ratio_
#    
#    z = np.zeros((60, 50))
#    for j in range(10,60):
#
#        algorithm = LocalOutlierFactor(n_neighbors=j, contamination=outliers_fraction)
#        y = algorithm.fit_predict(X)
#
#        z[:,j-10] = -algorithm.negative_outlier_factor_   
#    
#    z_max = np.max(z,1)
#    z_pred = np.int64(z_max<1.4)
#    
#    Z_qualified[:, i] = z_max
#    Z_pred_qualified[:, i] = z_pred
#    
#    #    ------------------------------------------------------
       

    #
    colors = np.array(['#377eb8', '#ff7f00'])
    ax.scatter(X_r[:, 0], X_r[:, 1], color = colors[(z_pred+1)//2])
    # ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], color = 'blue')
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    #
    # ax.set_zlabel('Z')
    ax.set_title('PCA outlier %s, outliers_fraction = %.2f' % (str(i+1),outliers_fraction))
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    # plt.savefig("figure/outlier/%s.png" % str(i+1))


#Z_diff=Z-Z_qualified
#Z_divide = Z/Z_qualified
    
plt.show()


