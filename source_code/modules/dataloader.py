import glob
import os

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


def get_data(path, ctype, extension="jpg", shape=(350,350), resolution=(20,20), as_gray=True, verbose=True):
    """Helper function to create consumable data from the set.
    : path       -> path of the folder where images are present
    : ctype      -> the type of the class the data belongs to (pos / neg)
    : extension  -> extension of the images (.jpg, .png)
    : shape      -> cropping shape of the initial image to capture the face only
    : resolution -> final resolution of the images like (40x40, 20x20, 60x60) 
    : as_gray    -> whether to convert to grayscale or not
    : verbose ->
    :
    data = {"id0": {"label":1,
                    "img": np.array(),
                    "ii_img": np.array(),
                    },
            "id1": {"label":1,
                    "img": np.array(),
                    "ii_img": np.array(),
                    },
            ...
            ...
            }
    """
    data = {}
    idx = 0

    for fname in glob.glob(f"{path}*.{extension}"):
        x_lb = shape[0]
        y_lb = shape[1]
        img = io.imread(fname, as_gray=as_gray)
        if img.shape[0]>=x_lb and img.shape[1]>=y_lb:
            # add the file names
            _id_ = f"{ctype}_{idx}"
            idx += 1
            data[_id_] = {"fname": fname}
            
            xin = np.int((img.shape[0]-x_lb)/2)
            xout = np.int((img.shape[0]+x_lb)/2)
            yin = np.int((img.shape[1]-y_lb)/2)
            yout = np.int((img.shape[1]+y_lb)/2)
            res = cv2.resize(img[xin:xout, yin:yout], dsize=resolution, interpolation=cv2.INTER_CUBIC)
            # res = skimage.transform.resize(img, resolution, anti_aliasing=True)
            data[_id_]["img"] = res

            ## add the labels
            if ctype == "pos":
                data[_id_]["label"] = 1
            elif ctype == "neg":
                data[_id_]["label"] = 0

    if verbose: print(f"num_data => {len(data.keys())} :: img_shape = {data[_id_]['img'].shape}")
    
    return data


def create_integral_image(img, do_plot=False):
    """Create integral_image:
    : img -> Image
    : do_plot -> Whether to plot the integral_image? or not?
    : Returns::
    : ii_img -> integral image
    """
    t0 = time.time()
    ii_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ii_img[i, j] = np.sum(img[:i+1, :j+1])

    ## padding the edges and corners with zeros (boundary conditions)
    tmp = np.zeros((ii_img.shape[0]+2, ii_img.shape[1]+2))
    tmp[1:-1, 1:-1] = ii_img
    ii_img = tmp

    if do_plot:
        plt.figure()
        plt.imshow(ii_img)

    return ii_img

