
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

import modules.utils as utils


def init_weights(data, keys, dist="Uniform"):
    """Initialize weights for the data
    : data ->
    : keys ->
    : dist ->
    """
    
    if dist == "Uniform":
        num_train = len(keys)
        for k in keys:
            data[k]["weight"] = 1. / num_train
    if dist == "XUniform":
        C = len(keys)                                   ## total number of classes
        for c_k in keys:
            num = len(c_k)                              ## total number of data items in each class
            for k in c_k:
                data[k]["weight"] = 1. / (C * num)


def calc_threshold_and_polarity(data, classifiers):
    """get threshold for each weak-classifier
    : data ->
    : classifiers ->
    """
    
    for c_id in classifiers:
        ## sort the feat_val for each image in ascending order of the feat_val
        sorted_vals = sorted(classifiers[c_id]["val"].items(), key=lambda x:x[1])
        ## calculate the weights of LEFT=(neg_till + pos_all - pos_till) & RIGHT=(pos_till + neg_all -neg_till)
        sum_weights_neg_all = 0
        sum_weights_pos_all = 0
        sum_weights_neg_till = 0
        sum_weights_pos_till = 0
        errs = {}
        polarity = {}
        for i_id in data:
            if data[i_id]["label"] == 0:
                sum_weights_neg_all += data[i_id]["weight"]
            elif data[i_id]["label"] == 1:
                sum_weights_pos_all += data[i_id]["weight"]

        for item in sorted_vals:
            ## item -> (i_id, f_val)
            i_id = item[0]
            if data[i_id]["label"] == 0:
                sum_weights_neg_till += data[i_id]["weight"]
            elif data[i_id]["label"] == 1:
                sum_weights_pos_till += data[i_id]["weight"]
            left = sum_weights_neg_till + sum_weights_pos_all - sum_weights_pos_till
            right = sum_weights_pos_till + sum_weights_neg_all - sum_weights_neg_till
            # errs[i_id] = min(left, right)
            if left <= right:
                errs[i_id] = left
                polarity[i_id] = -1
            elif left > right:
                errs[i_id] = right
                polarity[i_id] = 1
        
        thres_iid = sorted(errs.items(), key=lambda x:x[1])[0][0]
        classifiers[c_id]["threshold"] = classifiers[c_id]["val"][thres_iid]
        classifiers[c_id]["polarity"] = polarity[thres_iid]

def calc_pred_and_error(data, classifiers):
    for c_id in classifiers:
        ## initialize predicted_labels
        classifiers[c_id]["pred_labels"] = {}
        error = 0
        for i_id in classifiers[c_id]["val"]:
            polarity = classifiers[c_id]["polarity"]
            if polarity * classifiers[c_id]["val"][i_id] >= polarity * classifiers[c_id]["threshold"]:
                classifiers[c_id]["pred_labels"][i_id] = 1
            else:
                classifiers[c_id]["pred_labels"][i_id] = 0

            if classifiers[c_id]["pred_labels"][i_id] != data[i_id]["label"]:
                error += data[i_id]["weight"]
        classifiers[c_id]["error"] = error



def get_best_classifier(classifiers):
    """
    : classifiers ->
    """

    sorted_errors = sorted([(c_id, classifiers[c_id]["error"]) for c_id in classifiers], key=lambda x:x[1])
    # sorted_errors = sorted(error_all_feats.items(), key=lambda x:x[1])
    best_c_id = sorted_errors[0][0]
    best_error = sorted_errors[0][1]
    best_alpha = 0.5 * np.log((1-best_error)/best_error)
    classifiers[best_c_id]["alpha"] = best_alpha
    return best_c_id



def update_normalized_weights(data, classifiers, c_id):
    """
    : data ->
    : classifiers ->
    : c_id -> best_c_id
    """
    
    ## calculate the weight normalization factor
    norm_factor = 0
    for i_id in data:
        if classifiers[c_id]["pred_labels"][i_id] != data[i_id]["label"]:
            norm_factor += data[i_id]["weight"] * np.exp(classifiers[c_id]["alpha"])
        elif classifiers[c_id]["pred_labels"][i_id] == data[i_id]["label"]:
            norm_factor += data[i_id]["weight"] * np.exp(-1 * classifiers[c_id]["alpha"])

    ## normalize and update the weights of all data
    for i_id in data:        
        if classifiers[c_id]["pred_labels"][i_id] != data[i_id]["label"]:
            data[i_id]["weight"] *= np.exp(classifiers[c_id]["alpha"])
            data[i_id]["weight"] /= norm_factor
        elif classifiers[c_id]["pred_labels"][i_id] == data[i_id]["label"]:
            data[i_id]["weight"] *= np.exp(-1 * classifiers[c_id]["alpha"])
            data[i_id]["weight"] /= norm_factor



def run_adaboost(data, classifiers, num_epochs, verbose=True):
    """
    👍 for iter t <= T:
        👍 Calculate the vals for each image for each weak-classifier
        👍 Get threshold
        👍 Get error
        👍 Get the best-feature (weak-classifier)
        👍 Get the best-features's ALPHA
        👍 Update the weights with normalization
        👍 Return the (best-feature, ALPHA)
    : data ->
    : classifiers ->
    : num_steps ->
    : verbose ->
    """
    
    strong_classifier = []          ## a strong-classifier is an array of multiple weak-classifiers
    for epoch in range(num_epochs):
        cid0 = list(classifiers.keys())[0]
        if epoch==0 and (len(classifiers[cid0]["val"].keys()) == 0):
            ## calculate "val" only for the 1st epoch (as it remains same after)
            t0 = time.time()
            for c_id in classifiers:
                for i_id in data:
                    ## 👍 calculate the vals for all data
                    utils.calc_feat_val(data=data, classifiers=classifiers, i_id=i_id, c_id=c_id)
            print(f"⏰utils.calc_feat_val() took => {time.time()-t0} seconds @epoch[{epoch}]!!")
        ## 👍 get threshold and error for each weak-classifier
        calc_threshold_and_polarity(data, classifiers)
        calc_pred_and_error(data, classifiers)
        ## 👍 Get the best-feature (weak-classifier), its ALPHA & store it in strong-classifier
        best_c_id = get_best_classifier(classifiers)
        # strong_classifier[best_c_id] = classifiers[best_c_id].copy()
        strong_classifier.append({best_c_id: classifiers[best_c_id].copy()})
        # strong_classifier[t] = classifiers[best_c_id].copy()
        ## 👍 Normalize & update the weights
        update_normalized_weights(data=data, classifiers=classifiers, c_id=best_c_id)
        if verbose:
            print(f"epoch[{epoch}/{num_epochs}] => {best_c_id} " \
                f":: (error->{classifiers[best_c_id]['error']}) " \
                f":: (alpha->{classifiers[best_c_id]['alpha']})")
    return strong_classifier


class Cascade():
    def __init__(self, strong_classifier=None):
        self.strong_classifier = strong_classifier
    
    def train(self, data, keys, num_epochs, c_types, weight_dist="Uniform", template=(0,0,20,20), verbose=False):
        """
        : data ->
        : c_types -> different types of classifiers (haar-type-classifiers) :: ["haar_lr", "haar_tb", "haar_lcr", "haar_tcb", "haar_dd"]
        : weight_dist -> ::"Uniform" or "XUniform"
        : template -> (i0, j0, h, w) of the window  :: (0,0,20,20)
        : verbose -> :: True or False
        """
        self.num_epochs = num_epochs
        
        t0 = time.time()
        ## 👍 initialize the weights
        # keys = data.keys()
        init_weights(data=data, keys=keys, dist=weight_dist)

        ## 👍 create ALL haar-like-features
        classifiers = utils.create_weak_classifiers(ctypes=c_types, 
                                                   i0=template[0],
                                                   j0=template[1],
                                                   w=template[2],
                                                   h=template[3])
        
        ## write rules for training in each mode-type
        
        if verbose: print(f"<|CASCADE|>|> #classifiers=>{len(classifiers)}")
        self.strong_classifier = run_adaboost(data=data, classifiers=classifiers, num_epochs=self.num_epochs, verbose=verbose)
        if verbose: print(f"Time taken => {time.time()-t0} seconds!!")
        
    
    def test(self, data, mode, verbose=False):
        """
        : data ->
        : mode -> ::["sequential", "committe", "attentional"]
        : verbose ->
        """
        assert len(self.strong_classifier) != 0
        self.mode=mode
        if mode == "sequential":
            allowed_iids = list(data.keys())
            stage = 0
            for wc in self.strong_classifier:
                c_id = list(wc.keys())[0]
                wc[c_id]["val"] = {}           ## reset the garbage from training process
                wc[c_id]["pred_labels"] = {}   ## reset the garbage from training process
                error = 0
                new_allowed_iids = []
                
                for i_id in allowed_iids:
                    utils.calc_feat_val(data=data, classifiers=wc, i_id=i_id, c_id=c_id)
                    polarity = wc[c_id]["polarity"]
                    if polarity * wc[c_id]["val"][i_id] >= polarity * wc[c_id]["threshold"]:
                        wc[c_id]["pred_labels"][i_id] = 1
                        new_allowed_iids.append(i_id)             ## only faces are sent to the next weak-clasifier
                    else:
                        wc[c_id]["pred_labels"][i_id] = 0

                    if wc[c_id]["pred_labels"][i_id] != data[i_id]["label"]:
                        error += 1
                wc[c_id]["stage_test_error"] = error/len(allowed_iids)

                ## update the allowed_iids to be classified by next weak-classifier
                allowed_iids = new_allowed_iids

                if verbose: 
                    print(f"mode=[{mode}]: stage[{stage}]=> passed = [{len(allowed_iids)}/{len(data)}] " \
                        f":: stage_test_error -> {wc[c_id]['stage_test_error']} " \
                        f":: overall_test_error -> {error/len(data)}")
                stage += 1
                
        elif mode == "committee":
            ## if (sum_t(alpha_t * polarity_t * val_t_i) >= sum_t(alpha_t * polarity_t * thresh_t)) => 1 else 0
            pred_xvals = {}
            weighted_committe_threshold = 0
            error = 0
            for wc in self.strong_classifier:
                c_id = list(wc.keys())[0]
                wc[c_id]["val"] = {}           ## reset the garbage from training process
                wc[c_id]["pred_labels"] = {}   ## reset the garbage from training process
                weighted_committe_threshold += wc[c_id]["alpha"] * wc[c_id]["polarity"] * wc[c_id]["threshold"]
                
                for i_id in data.keys():
                    utils.calc_feat_val(data=data, classifiers=wc, i_id=i_id, c_id=c_id)
                    if i_id not in pred_xvals.keys():
                        pred_xvals[i_id] = [wc[c_id]["alpha"] * wc[c_id]["polarity"] * wc[c_id]["val"][i_id]]
                    else:
                        pred_xvals[i_id].append(wc[c_id]["alpha"] * wc[c_id]["polarity"] * wc[c_id]["val"][i_id])
            
            for i_id in data.keys():
                pred_xvals[i_id].append(np.sum(pred_xvals[i_id]))
                
                ## calculate the predicted labels
                if pred_xvals[i_id][-1] >= weighted_committe_threshold:
                    pred_xvals[i_id].append(1)
                    if pred_xvals[i_id][-1] != data[i_id]["label"]: error += 1
                else:
                    pred_xvals[i_id].append(0)
                    if pred_xvals[i_id][-1] != data[i_id]["label"]: error += 1
                    
            ## calculate the "error" for the commitee
            error /= len(data.keys())
            if verbose: 
                    print(f"mode=[{mode}] => overall_test_error -> {error}")
            res = {}
            for iid in pred_xvals.keys():
                res[iid] = pred_xvals[iid][-2]
            return res

    def draw(self, data, i_id, win_h, win_w):
        """
        : data -> used to draw the background overlay
        : i_id -> which image to draw on the background of the classifiers 
        : win_h -> window width
        : win_w -> window height
        """
        utils.draw_classifier(data=data, strong_classifier=self.strong_classifier, win_h=win_h, win_w=win_w, i_id=i_id)
