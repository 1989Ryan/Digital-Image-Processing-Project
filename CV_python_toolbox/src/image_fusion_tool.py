import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
from random import *
from PIL import Image
from sklearn.decomposition import PCA
from src.basic_cv_tool import *

class image_fusion_tool:
    
    def __init__(self, ImageName):
        self.ImageName = ImageName
    
    def weighted_average_fusion(self, img1, img2, w1, w2):
        '''
        This is the simplest image fusion algorithm. 
        :param img1: The first origin image.
        :param img2: The second origin image.
        :param w1: The weight of first image.
        :param w2: The weight of second image.
        :return: The fusioned image.
        '''
        if w1<0 or w2<0:
            print('invalid weight value')
            return
        elif w1 + w2 != 1:
            w1 = w1/(w1+w2)
            w2 = w2/(w1+w2)
        shape = np.shape(img1)
        img = np.zeros(shape,dtype = np.int8)
        if np.shape(img2) != shape:
            img2 = cv2.resize(img2, np.shape(img1), interpolation = cv2.INTER_CUBIC)
        img = w1*img1+w2*img2
        return img
    
    def PCA_image_fusion(self, img1, img2):
        '''
        This is the algorithm of image fusion based on PCA.
        :param img1: The origin image.
        :param img2: The high resolution image.
        :return: The fusioned image.
        '''
        estimator = PCA()
        estimator.fit(img1.copy())
        estimator.fit(img2.copy())
        img_f1 = estimator.transform(img1.copy())
        img_f2 = estimator.transform(img2.copy())
        img_f1[:,:40] = img_f2[:,:40]
        img = estimator.inverse_transform(img_f1)
        return img
    
    def HSI_image_fusion(self, img1, img2):
        '''
        :param img1: The origin image.
        :param img2: The high resolution image.
        :return: The fusioned image.
        '''
        tool = basic_cv_tool('')
        hsi_img1 = tool.RGB2HSI(img1)
        hsi_img2 = tool.RGB2HSI(img2)
        hsi_img1[:,:,2] = hsi_img2[:,:,2]
        img = tool.HSI2RGB(hsi_img1)
        return img
        