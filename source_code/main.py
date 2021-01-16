
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

import modules.dataloader as dl
import modules.adaboost as adaboost
import modules.utils as utils

phase = "train"        ## <- toggle it to "test" to run the testing module

path_pos_data_train = f"../data/face_train/"
path_pos_data_train = f"../data/face_train/"

path_pos_data_test = f"../data/face_test/"
path_neg_data_test = f"../data/bg_test/"

shape=(350, 350)
resolution=(20, 20)

if phase=="train":
    ## 🎉🎉 MASTER CODE 🎉🎉 PART-1 (TRAIN DATALOADING) 🎉🎉
    ## 
    ## 👍 grab the pos & neg training data (windows) and their labels
    print(f"Training!!")

    pos_data_train = dl.get_data(path=path_pos_data_train, shape=shape, resolution=resolution, as_gray=True, ctype="pos")
    neg_data_train = dl.get_data(path=path_neg_data_train, shape=(20, 20), resolution=resolution, as_gray=True, ctype="neg")
    data_train = {**pos_data_train, **neg_data_train}

    ## 🥵 calculate integral-image for all data
    for i_id in data_train.keys():
        ii_img = dl.create_integral_image(data_train[i_id]["img"])
        data_train[i_id]["ii_img"] = ii_img

    ## adaboost.Cascade
    ## 🚀🚀 MASTER CODE 🚀🚀 PART-2.new (CASCADE training) 🚀🚀
    ## 
    num_epochs = 5
    c_types = ["haar_lr", "haar_lcr", "haar_tb", "haar_tcb", "haar_dd"]
    weight_dist = "Uniform"
    # i_id_keys = [list(pos_data_train), list(neg_data_train)]
    i_id_keys = data_train.keys()
    template = (0,0,20,20)
    mode = "sequential"   ## otherwise can use these aswell -> ["committe", "attentional"]

    ## Build & train a cascade
    cascade = adaboost.Cascade()
    cascade.train(data=data_train, keys=i_id_keys, num_epochs=num_epochs, c_types=c_types, weight_dist=weight_dist, template=template, verbose=True)
        

elif phase=="test":
    print(f"Testing!!")
    ## 🚀🚀 MASTER CODE 🚀🚀 PART-3 (TEST DATA-LOADING) 🚀🚀
    ## 
    ## 👍 grab the pos & neg training data (windows) and their labels

    pos_data_test = dl.get_data(path=path_pos_data_test, shape=shape, resolution=resolution, as_gray=True, ctype="pos")
    neg_data_test = dl.get_data(path=path_neg_data_test, shape=(20,20), resolution=resolution, as_gray=True, ctype="neg")
    data_test = {**pos_data_test, **neg_data_test}

    ## 🥵 calculate integral-image for all data
    for i_id in data_test.keys():
        ii_img = dl.create_integral_image(data_test[i_id]["img"])
        data_test[i_id]["ii_img"] = ii_img

    ## 🚀🚀 MASTER CODE 🚀🚀 PART-4 (TEST using trained CASCADE) 🚀🚀
    ## 
    # sc = cascade.strong_classifier              ## <- 🚫CAREFUL: used only for speed-testing to avoid re-training. DONOT include in final_source
    # cascade = adaboost.Cascade(strong_classifier=sc)     ## <- 🚫CAREFUL: used only for speed-testing to avoid re-training. DONOT include in final_source

    cascade.test(data=data_test, mode="sequential", verbose=True)
    # predictions = cascade.test(data=data_test, mode="committee", verbose=True)

    cascade.draw(data_test, "pos_0", 20, 20)

    utils.draw_roc_curve(data=data_test, mode="sequential", classifiers=cascade.strong_classifier)
    # utils.draw_roc_curve(data=data_test, mode="committee", predictions=predictions)
