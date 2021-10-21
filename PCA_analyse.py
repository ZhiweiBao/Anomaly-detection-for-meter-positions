
import numpy as np


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor



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

n_r = range(5,51,5)
n_comp = list(n_r)

evr_sum = np.zeros((20, len(n_comp)))


for j in n_r:
    evr = np.zeros((j, 20))
    for i in range(20):
    
        X = dataset[i]
        if np.isnan(X.min()):
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X)
            X = imp.transform(X)
    
        pt = PowerTransformer()
        X = pt.fit(X).transform(X)
    
#        qt = QuantileTransformer(output_distribution='normal')
#        X = qt.fit_transform(X)
        
    #    X = Normalizer().fit_transform(X)
#        X = StandardScaler().fit_transform(X)
#        X = RobustScaler().fit_transform(X)
        
        pca = PCA(n_components=j, whiten=True)
        X_r = pca.fit(X).transform(X)
        evr[:, i] = pca.explained_variance_ratio_
        
    
        algorithm = LocalOutlierFactor(contamination=outliers_fraction)
        y = algorithm.fit_predict(X)
        y_pred = algorithm.fit_predict(X_r)
        z = algorithm.negative_outlier_factor_
        
        z_pred = np.int64(z>-1.5)
        
        Y[:, i] = y
        Z[:, i] = z
        Z_pred[:, i] = z_pred
        
    evr_sum[:,j//5-1] = np.sum(evr,0)


evr_sum = 100 * evr_sum



