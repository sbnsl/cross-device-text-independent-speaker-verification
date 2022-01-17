import numpy as np

from sklearn import *
import matplotlib.pyplot as plt

def calculate_eer_auc(label, distance,plot = False):
    # neg_distance =  -1*distance
    #print neg_distance
    fpr, tpr, thresholds = metrics.roc_curve(label, -distance, pos_label=1)
    AUC = metrics.roc_auc_score(label,-distance, average='macro', sample_weight=None)
    # Calculating EER
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    EER = intersect_x

    if plot:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return EER,AUC

