import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
import evaluation

n_clusters = 3
data = pd.read_csv('../../data/Survival_data/5BRCA/lowDim=20_alpha=1_gamma=10_X.csv', header=None)
data = np.array(data)
for i in range(3):
    km = KMeans(n_clusters=n_clusters + i, init='k-means++')
    y_pred = km.fit_predict(data)
    np.savetxt('../../data/Survival_data/5BRCA/defusion_cluster_label_{number}.txt'.format(number=n_clusters + i), y_pred)
    y_pred = pd.DataFrame(y_pred)
    file = open('../../data/Survival_data/5BRCA/defusion_cluster_label_{number}.csv'.format(number=n_clusters + i), 'w')
    y_pred.to_csv(file, header=0, index=None)