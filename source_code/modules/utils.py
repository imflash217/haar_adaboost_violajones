
####################################################################################################
##### NC State University | Spring 2020
##### Course  : ECE-763 Computer Vision 
##### PROJECT #2 (Haar features, AdaBoost Algorithm, Viola-Jones Face Detection)
#####...............................................................................................
##### Author  : Vinay Kumar
##### UnityID : vkumar24@ncsu.edu
#####...............................................................................................
##### This is a work of © vkumar24@ncsu.edu
####################################################################################################

import os
import pprint
import time
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage import io
import skimage
import cv2
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


import modules.haar_features as hf

def create_weak_classifiers(ctypes, i0, j0, w, h):
    """
    : ctypes ->
    : i0 ->
    : j0 ->
    : w ->
    : h ->
    """
    classifiers = {}
    for c in ctypes:
        # print(c)
        if c == "haar_lr":
            h_lr = hf.HaarTypeLR(i0, j0, w, h)
            h_lr.init_feats()
            classifiers.update(h_lr.feats)
        elif c == "haar_lcr":
            h_lcr = hf.HaarTypeLCR(i0, j0, w, h)
            h_lcr.init_feats()
            classifiers.update(h_lcr.feats)
        elif c == "haar_tb":
            h_tb = hf.HaarTypeTB(i0, j0, w, h)
            h_tb.init_feats()
            classifiers.update(h_tb.feats)
        elif c == "haar_tcb":
            h_tcb = hf.HaarTypeTCB(i0, j0, w, h)
            h_tcb.init_feats()
            classifiers.update(h_tcb.feats)
        elif c == "haar_dd":
            h_dd = hf.HaarTypeDD(i0, j0, w, h)
            h_dd.init_feats()
            classifiers.update(h_dd.feats)
    return classifiers
    
def calc_feat_val(data, classifiers, i_id, c_id):
    """
    : data ->
    : classifiers ->
    : i_id -> index(ID) of the image
    : c_id -> index(ID) of the weak-classifier => ((i0,j0,h,w), ctype)
    """
    if classifiers[c_id]["feat_type"] == "haar_lr" or classifiers[c_id]["feat_type"] == "haar_tb":
        sum_A = hf.get_sum_pixels(coord=classifiers[c_id]["coord_A"], ii_img=data[i_id]["ii_img"])
        sum_B = hf.get_sum_pixels(coord=classifiers[c_id]["coord_B"], ii_img=data[i_id]["ii_img"])
        val = sum_A - sum_B
        classifiers[c_id]["val"][i_id] = val
        
    elif classifiers[c_id]["feat_type"] == "haar_lcr" or classifiers[c_id]["feat_type"] == "haar_tcb":
        sum_A = hf.get_sum_pixels(coord=classifiers[c_id]["coord_A"], ii_img=data[i_id]["ii_img"])
        sum_B = hf.get_sum_pixels(coord=classifiers[c_id]["coord_B"], ii_img=data[i_id]["ii_img"])
        sum_C = hf.get_sum_pixels(coord=classifiers[c_id]["coord_C"], ii_img=data[i_id]["ii_img"])
        val = sum_A - sum_B + sum_C
        classifiers[c_id]["val"][i_id] = val
        
    elif classifiers[c_id]["feat_type"] == "haar_dd":
        sum_A = hf.get_sum_pixels(coord=classifiers[c_id]["coord_A"], ii_img=data[i_id]["ii_img"])
        sum_B = hf.get_sum_pixels(coord=classifiers[c_id]["coord_B"], ii_img=data[i_id]["ii_img"])
        sum_C = hf.get_sum_pixels(coord=classifiers[c_id]["coord_C"], ii_img=data[i_id]["ii_img"])
        sum_D = hf.get_sum_pixels(coord=classifiers[c_id]["coord_D"], ii_img=data[i_id]["ii_img"])
        val = sum_A - sum_B + sum_C - sum_D
        classifiers[c_id]["val"][i_id] = val


def draw_classifier(data, strong_classifier, win_h, win_w, i_id):
    """
    : data ->
    : strong_classifier -> 
    : win_h ->
    : win_w ->
    : i_id ->
    """

    for wc in strong_classifier:
        c_id = list(wc.keys())[0]
        classifier_img = np.ones((win_h+2, win_w+2))*255
        # print(classifier_img)

        i0 = c_id[0][0]
        j0 = c_id[0][1]
        h = c_id[0][2]
        w = c_id[0][3]
        c_type = c_id[1]
        if c_type=="haar_lr":
            for i in range(i0, i0+h):
                for j in range(j0, j0+w):
                    classifier_img[i,j] = 125          ## A
                for j in range(j0+w, j0+2*w):
                    classifier_img[i,j] = 0            ## B
        elif c_type=="haar_lcr":
            for i in range(i0, i0+h):
                for j in range(j0, j0+w):
                    classifier_img[i,j] = 125          ## A
                for j in range(j0+w, j0+2*w):
                    classifier_img[i,j] = 0            ## B
                for j in range(j0+2*w, j0+3*w):
                    classifier_img[i,j] = 125          ## C
        elif c_type=="haar_tb":
            for j in range(j0, j0+w):
                for i in range(i0, i0+h):
                    classifier_img[i,j] = 125          ## A
                for i in range(i0+h, i0+2*h):
                    classifier_img[i,j] = 0            ## B
        elif c_type=="haar_tcb":
            for j in range(j0, j0+w):
                for i in range(i0, i0+h):
                    classifier_img[i,j] = 125          ## A
                for i in range(i0+h, i0+2*h):
                    classifier_img[i,j] = 0            ## B
                for i in range(i0+2*h, i0+3*h):
                    classifier_img[i,j] = 125          ## C
        elif c_type=="haar_dd":
            for i in range(i0, i0+h):
                for j in range(j0, j0+w):
                    classifier_img[i,j] = 125          ## A
                for j in range(j0+w, j0+2*w):
                    classifier_img[i,j] = 0            ## B
            for i in range(i0+h, i0+2*h):
                for j in range(j0, j0+w):
                    classifier_img[i,j] = 0            ## C
                for j in range(j0+w, j0+2*w):
                    classifier_img[i,j] = 125          ## D

        plt.figure()
        plt.imshow(data[i_id]["img"], cmap="gray")
        plt.imshow(classifier_img[1:-1, 1:-1], cmap="hot", alpha=0.7)
        plt.title(f"{c_id}, error={wc[c_id]['error']}, alpha={wc[c_id]['alpha']}")
    

def draw_roc_curve(data, mode, classifiers=None, predictions=None):
    """Draw ROC curve & AUC
    : classifiers -> [optional]
    : predictions -> [optional]
    """
    plt.figure()
    plt.title("ROC Curve for all classifiers")
    
    if mode == "sequential":
        idx = 0
        for wc in classifiers:
            c_id = list(wc.keys())[0]
            polarity = wc[c_id]["polarity"]
            iids = sorted(list(wc[c_id]["val"].keys()))
            labels = []
            preds_vals = []
            val_min = 1e6
            val_max = -1e6
            for i_id in iids:
                labels.append(data[i_id]["label"])
                val = wc[c_id]["val"][i_id]
                preds_vals.append(val)
                if val_min > val:
                    val_min = val
                if val_max < val:
                    val_max = val

            D = np.zeros((100, len(iids)+6))
            for i, t in enumerate(np.linspace(val_min, val_max, 100)):
                preds = [int(polarity*p > polarity*t) for p in preds_vals]
                D[i][:-6] = preds

                for j in range(len(D[i])-6):
                    if labels[j] == 1 and D[i, j] == 1:   ## TP
                        D[i,-6] += 1
                    if labels[j] == 1 and D[i, j] == 0:   ## FP
                        D[i,-5] += 1
                    if labels[j] == 0 and D[i, j] == 0:   ## TN
                        D[i,-4] += 1
                    if labels[j] == 0 and D[i, j] == 1:   ## FN
                        D[i,-3] += 1
                if (D[i,-6] + D[i,-3]) != 0:
                    D[i][-2] = D[i,-6] / (D[i,-6] + D[i,-3])   ## TPR
                if (D[i,-5] + D[i,-4]) != 0:
                    D[i][-1] = D[i,-5] / (D[i,-5] + D[i,-4])   ## FPR

            tpr = list(D[:, -2])
            fpr = list(D[:, -1])
            rates = [[f, t] for f, t in zip(fpr, tpr)]
            sorted_rates = np.array(sorted(rates, key=lambda x:x[0])).T

            auc = metrics.auc(sorted_rates[0], sorted_rates[1])

            plt.plot(sorted_rates[0], sorted_rates[1], label=f"stage-{idx}: AUC={auc}")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            idx += 1
    elif mode=="committee":
        # predictions -> {i_id: pred_xvals}
        labels = []
        iids = list(sorted(predictions.keys()))
        preds_xvals = []
        val_min = 1e6
        val_max = -1e6
        D = np.zeros((100, len(iids)+6))
        
        for i_id in iids:
            labels.append(data[i_id]["label"])
            xval = predictions[i_id]
            preds_xvals.append(xval)
            if val_min > xval:
                val_min = xval
            if val_max < xval:
                val_max = xval
        for i, t in enumerate(np.linspace(val_min, val_max, 100)):
            preds = [int(p >= t) for p in preds_xvals]
            D[i][:-6] = preds

            for j in range(len(D[i])-6):
                if labels[j] == 1 and D[i, j] == 1:   ## TP
                    D[i,-6] += 1
                if labels[j] == 1 and D[i, j] == 0:   ## FP
                    D[i,-5] += 1
                if labels[j] == 0 and D[i, j] == 0:   ## TN
                    D[i,-4] += 1
                if labels[j] == 0 and D[i, j] == 1:   ## FN
                    D[i,-3] += 1
            if (D[i,-6] + D[i,-3]) != 0:
                D[i][-2] = D[i,-6] / (D[i,-6] + D[i,-3])   ## TPR
            if (D[i,-5] + D[i,-4]) != 0:
                D[i][-1] = D[i,-5] / (D[i,-5] + D[i,-4])   ## FPR
        tpr = list(D[:, -2])
        fpr = list(D[:, -1])
        rates = [[f, t] for f, t in zip(fpr, tpr)]
        sorted_rates = np.array(sorted(rates, key=lambda x:x[0])).T

        auc = metrics.auc(sorted_rates[0], sorted_rates[1])

        plt.plot(sorted_rates[0], sorted_rates[1], label=f"{mode}: AUC={auc}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
    plt.show()

 