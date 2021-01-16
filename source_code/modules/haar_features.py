
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


def get_coordinates(coordinates):
    """
    : coordinates ->
    """

    # print("\n:::", coordinates)
    coord = coordinates.copy()
    for k in range(len(coord)):
        if coord[k][0] == coord[k][1]:
            coord[k].pop(0)

    if coord[0] == coord[1]:
        coord.pop(0)

    coord = np.array(coord)
    # print("\n::", coord)

    return coord


def get_sum_pixels(coord, ii_img):
    """Getting the sum of all pixels in a given coordinate array and the integral_image
    : coord ->
    : ii_img ->
    """

    if coord.shape[0]==1 and coord.shape[1]==1:
        A = ii_img[coord[0,0,0]-1, coord[0,0,1]-1]
        B = ii_img[coord[0,0,0]-1, coord[0,0,1]]
        C = ii_img[coord[0,0,0], coord[0,0,1]-1]
        D = ii_img[coord[0,0,0], coord[0,0,1]]
    if coord.shape[0]==1 and coord.shape[1]!=1:
        A = ii_img[coord[0,0,0]-1, coord[0,0,1]-1]
        B = ii_img[coord[0,1,0]-1, coord[0,1,1]]
        C = ii_img[coord[0,0,0], coord[0,0,1]-1]
        D = ii_img[coord[0,1,0], coord[0,1,1]]
    if coord.shape[0]!=1 and coord.shape[1]==1:
        A = ii_img[coord[0,0,0]-1, coord[0,0,1]-1]
        B = ii_img[coord[0,0,0]-1, coord[0,0,1]]
        C = ii_img[coord[1,0,0], coord[1,0,1]-1]
        D = ii_img[coord[1,0,0], coord[1,0,1]]
    if coord.shape[0]!=1 and coord.shape[1]!=1:
        A = ii_img[coord[0,0,0]-1, coord[0,0,1]-1]
        B = ii_img[coord[0,1,0]-1, coord[0,1,1]]
        C = ii_img[coord[1,0,0], coord[1,0,1]-1]
        D = ii_img[coord[1,1,0], coord[1,1,1]]
    
    # print(A, B, C, D, "\n-----------\n")

    ## calculate the feature value using the integral image => (A-B-C+D)
    sum_pixels = A-B-C+D
    return sum_pixels


class HaarFeatures():
    """
    Create discriptions for all the different types of haar-like-features
    """

    def __init__(self, win_i0, win_j0, win_w, win_h):
        """
        : win_i0 ->
        : win_j0 ->
        : win_w ->
        : win_h ->
        """
        self.feat_type = "default"
        self.win_i0 = win_i0
        self.win_j0 = win_j0
        self.win_w = win_w
        self.win_h = win_h
        self.feats = dict()

    def _init_feats(self):
        pass

    def init_feats(self):
        self._init_feats()



class HaarTypeLR(HaarFeatures):
    """
    Types of haar_features extracted are displayed below:
    ______
    |*||#|
    ------
    ________
    |**||##|
    --------
    ________
    |*||#|
    |*||#|
    --------
    ____________
    |****||####|
    ------------
    ____________
    |****||####|
    |****||####|
    ------------
    and so on ...
    """
    
    def __init__(self, win_i0, win_j0, win_h, win_w):
        """
        docstring
        """
        super().__init__(win_i0, win_j0, win_h, win_w)
        self.feat_type = "haar_lr"

    
    def _init_feats(self):
        for i in range(self.win_i0+1, self.win_h):
            for j in range(self.win_j0+1, self.win_w):
                h = 1
                while i+h <= self.win_h:
                    w = 1
                    while j+2*w <= self.win_w:
                        coords = []
                        coord_A = [[(i, j), (i, j+w-1)], [(i+h-1, j), (i+h-1, j+w-1)]]
                        coord_B = [[(i, j+w), (i, j+2*w-1)], [(i+h-1, j+w), (i+h-1, j+2*w-1)]]

                        coords.append(coord_A)
                        coords.append(coord_B)

                        cx = []
                        for t in range(len(coords)):
                            c = get_coordinates(coords[t])
                            cx.append(c)
                        
                        self.feats[((i,j,h,w), self.feat_type)] = {"feat_type": self.feat_type, 
                                                                   "val": {}, 
                                                                   "error": 0.,
                                                                   "alpha": 0.,
                                                                   "coord_A": cx[0], 
                                                                   "coord_B": cx[1],
                                                                   }
                        
                        cx = []     ## reset the coordinates for next round
                        w += 1      ## increment the weight
                    h += 1          ## increment the height
   

class HaarTypeLCR(HaarFeatures):
    """
    Types of haar_features extracted are displayed below:
    _________
    |*||#||*|
    ---------
    ___________
    |**||##||**|
    -----------
    _________
    |*||#||*|
    |*||#||*|
    ---------
    __________________
    |****||####||****|
    ------------------
    __________________
    |****||####||****|
    |****||####||****|
    ------------------
    and so on ...
    """
    
    def __init__(self, win_i0, win_j0, win_h, win_w):
        """
        docstring
        """
        super().__init__(win_i0, win_j0, win_h, win_w)
        self.feat_type = "haar_lcr"

    
    def _init_feats(self):
        for i in range(self.win_i0+1, self.win_h):
            for j in range(self.win_j0+1, self.win_w):
                h = 1
                while i+h <= self.win_h:
                    w = 1
                    while j+3*w <= self.win_w:
                        coords = []
                        coord_A = [[(i, j), (i, j+w-1)], [(i+h-1, j), (i+h-1, j+w-1)]]
                        coord_B = [[(i, j+w), (i, j+2*w-1)], [(i+h-1, j+w), (i+h-1, j+2*w-1)]]
                        coord_C = [[(i, j+2*w), (i, j+3*w-1)], [(i+h-1, j+2*w), (i+h-1, j+3*w-1)]]

                        coords.append(coord_A)
                        coords.append(coord_B)
                        coords.append(coord_C)

                        cx = []
                        for t in range(len(coords)):
                            c = get_coordinates(coords[t])
                            cx.append(c)
                        
                        self.feats[((i,j,h,w), self.feat_type)] = {"feat_type": self.feat_type, 
                                                                   "val": {}, 
                                                                   "error": 0.,
                                                                   "alpha": 0.,
                                                                   "coord_A": cx[0], 
                                                                   "coord_B": cx[1],
                                                                   "coord_C": cx[2],
                                                                   }
                        
                        cx = []     ## reset the coordinates for next round
                        w += 1      ## increment the weight
                    h += 1          ## increment the height
   
        


class HaarTypeTB(HaarFeatures):
    """
    Types of haar_features extracted are displayed below:
    ___
    |*|
    |#|
    ---
    ___
    |*|
    |*|
    |#|
    |#|
    ---
    ____
    |**|
    |##|
    ----
    ____
    |**|
    |**|
    |##|
    |##|
    ----
    _________
    |*******|
    |#######|
    ---------
    _________
    |*******|
    |*******|
    |#######|
    |#######|
    ---------
    and so on ...
    """

    def __init__(self, win_i0, win_j0, win_w, win_h):
        """
        docstring
        """
        super().__init__(win_i0, win_j0, win_w, win_h)
        self.feat_type = "haar_tb"



    def _init_feats(self):
        for i in range(self.win_i0+1, self.win_h):
            for j in range(self.win_j0+1, self.win_w):
                h = 1
                while i+2*h <= self.win_h:
                    w = 1
                    while j+w <= self.win_w:
                        coords = []
                        coord_A = [[(i, j), (i, j+w-1)], [(i+h-1, j), (i+h-1, j+w-1)]]
                        coord_B = [[(i+h, j), (i+h, j+w-1)], [(i+2*h-1, j), (i+2*h-1, j+w-1)]]

                        coords.append(coord_A)
                        coords.append(coord_B)

                        cx = []
                        for t in range(len(coords)):
                            c = get_coordinates(coords[t])
                            cx.append(c)
                        
                        self.feats[((i,j,h,w), self.feat_type)] = {"feat_type": self.feat_type,
                                                                   "val": {}, 
                                                                   "error": 0.,
                                                                   "alpha": 0.,
                                                                   "coord_A": cx[0], 
                                                                   "coord_B": cx[1],
                                                                   }
                        
                        cx = []     ## reset the coordinates for next round
                        w += 1      ## increment the weight
                    h += 1          ## increment the height




class HaarTypeTCB(HaarFeatures):
    """
    Types of haar_features extracted are displayed below:
    ___
    |*|
    |#|
    |#|
    ---
    ___
    |*|
    |*|
    |#|
    |#|
    |*|
    |*|
    ---
    ____
    |**|
    |##|
    |##|
    ----
    ____
    |**|
    |**|
    |##|
    |##|
    |**|
    |**|
    ----
    _________
    |*******|
    |#######|
    |*******|
    ---------
    _________
    |*******|
    |*******|
    |#######|
    |#######|
    |*******|
    |*******|
    ---------
    and so on ...
    """

    def __init__(self, win_i0, win_j0, win_w, win_h):
        """
        docstring
        """
        super().__init__(win_i0, win_j0, win_w, win_h)
        self.feat_type = "haar_tcb"



    def _init_feats(self):
        for i in range(self.win_i0+1, self.win_h):
            for j in range(self.win_j0+1, self.win_w):
                h = 1
                while i+3*h <= self.win_h:
                    w = 1
                    while j+w <= self.win_w:
                        coords = []
                        coord_A = [[(i, j), (i, j+w-1)], [(i+h-1, j), (i+h-1, j+w-1)]]
                        coord_B = [[(i+h, j), (i+h, j+w-1)], [(i+2*h-1, j), (i+2*h-1, j+w-1)]]
                        coord_C = [[(i+2*h, j), (i+2*h, j+w-1)], [(i+3*h-1, j), (i+3*h-1, j+w-1)]]
                        

                        coords.append(coord_A)
                        coords.append(coord_B)
                        coords.append(coord_C)

                        cx = []
                        for t in range(len(coords)):
                            c = get_coordinates(coords[t])
                            cx.append(c)
                        
                        self.feats[((i,j,h,w), self.feat_type)] = {"feat_type": self.feat_type,
                                                                   "val": {}, 
                                                                   "error": 0.,
                                                                   "alpha": 0.,
                                                                   "coord_A": cx[0], 
                                                                   "coord_B": cx[1], 
                                                                   "coord_C": cx[2],
                                                                   }
                        
                        cx = []     ## reset the coordinates for next round
                        w += 1      ## increment the weight
                    h += 1          ## increment the height

## |>UNDER_CONSTRUCTION<|


class HaarTypeDD(HaarFeatures):
    """
    |>UNDER_CONSTRUCTION<|
    |>Define the rules for changing the h, w <|
    Types of haar_features extracted are displayed below:
    _____
    |*|#|
    |#|*|
    -----
    _____
    |*|#|
    |*|#|
    |#|*|
    |#|*|
    -----
    _______
    |**|##|
    |##|**|
    -------
    _______
    |**|##|
    |**|##|
    |##|**|
    |##|**|
    -------
    and so on ...
    """

    def __init__(self, win_i0, win_j0, win_w, win_h):
        """
        docstring
        """
        super().__init__(win_i0, win_j0, win_w, win_h)
        self.feat_type = "haar_dd"


    def _init_feats(self):
        for i in range(self.win_i0+1, self.win_h):
            for j in range(self.win_j0+1, self.win_w):
                h = 1
                while i+2*h <= self.win_h:
                    w = 1
                    while j+2*w <= self.win_w:
                        coords = []
                        coord_A = [[(i, j), (i, j+w-1)], [(i+h-1, j), (i+h-1, j+w-1)]]
                        coord_B = [[(i, j+w), (i, j+2*w-1)], [(i+h, j+w), (i+h, j+2*w-1)]]
                        coord_C = [[(i+h, j), (i+h, j+w-1)], [(i+2*h-1, j), (i+2*h-1, j+w-1)]]
                        coord_D = [[(i+h, j+w), (i+h, j+2*w-1)], [(i+2*h-1, j+w), (i+2*h-1, j+2*w-1)]]

                        coords.append(coord_A)
                        coords.append(coord_B)
                        coords.append(coord_C)
                        coords.append(coord_D)

                        cx = []
                        for t in range(len(coords)):
                            c = get_coordinates(coords[t])
                            cx.append(c)
                        
                        self.feats[((i,j,h,w), self.feat_type)] = {"feat_type": self.feat_type,
                                                                   "val": {}, 
                                                                   "error": 0.,
                                                                   "alpha": 0.,
                                                                   "coord_A": cx[0], 
                                                                   "coord_B": cx[1], 
                                                                   "coord_C": cx[2],
                                                                   "coord_D": cx[3],
                                                                   }
                        
                        cx = []     ## reset the coordinates for next round
                        w += 1      ## increment the weight
                    h += 1          ## increment the height
