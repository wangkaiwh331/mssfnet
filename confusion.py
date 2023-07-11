import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import os
import scipy.io as sio
import re
import csv




def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_test, y_pred, name):
    target_names = ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10', '_11', '_12', '_13', '_14',
                    '_15', '_16', '_17', '_18', '_19', '_20']

    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100

def getlabel_wk(y,removeZeroLabels = True):
    num=0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]>0:
                num=num+1
    patchesLabels = np.zeros((num))
    patchIndex = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r,c]>0:
                patchesLabels[patchIndex] = y[r, c]
                patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesLabels

def getpred_wk(pred,y,removeZeroLabels = True):
    num=0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]>0:
                num=num+1
    patchesLabels = np.zeros((num))
    patchespreds = np.zeros((num))
    patchIndex = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r,c]>0:
                patchesLabels[patchIndex] = y[r, c]
                patchespreds[patchIndex] = pred[r, c]
                patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesLabels = patchesLabels[patchesLabels>0]
        patchespreds = patchespreds[patchespreds>0]
        patchesLabels -= 1
        patchespreds -= 1
    return patchesLabels,patchespreds

def save_ndarray_as_csv(filename, array):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array)

class get_confusion():
    def __init__(self, true, pred,file_name,csv_name):
        super(get_confusion, self).__init__()
        classification, confusion, oa, each_acc, aa, kappa = reports(true, pred, 0)
        classification = str(classification)
        save_ndarray_as_csv(csv_name, confusion)
        confusion = str(confusion)

        with open(file_name, 'w') as x_file:
            x_file.write('{} '.format(kappa))
            x_file.write('\n')
            x_file.write('{} '.format(oa))
            x_file.write('\n')
            x_file.write('{} '.format(aa))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))



