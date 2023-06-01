import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
import evaluation


nmi = normalized_mutual_info_score
ari = adjusted_rand_score

true_label = pd.read_csv('../../data/SARC/labels_all_174.csv', header=None)
pred_label = pd.read_csv('../../data/SARC/nemo_pred.csv', header=None)

y_true = np.array(true_label).flatten()
y_pred = np.array(pred_label).flatten()

nmi, ari, f_score, acc = evaluation.evaluate(y_true, y_pred)

# print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f  <==|' % (nmi(y_true, y_pred), ari(y_true, y_pred)))
print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi, ari, f_score, acc))