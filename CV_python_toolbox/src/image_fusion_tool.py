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
    
    def gaussian_pyramid(self, img, level):
        temp = img.copy()
        pyramid_img = []
        for i in range(level):
            dst = cv2.pyrDown(temp)
            pyramid_img.append(dst)
            temp = dst.copy()
        return pyramid_img
    
    def laplacian_pyramid(self, img, level):
        pyramid_img = self.gaussian_pyramid(img, level)
        pyramid_lpls = []
        for i in range(level-1, -1, -1):
            if i-1<0:
                expend = cv2.pyrUp(pyramid_img[i], dstsize = img.shape[:2])
                lpls = cv2.subtract(img, expend)
                pyramid_lpls.append(lpls)
            else:
                expend = cv2.pyrUp(pyramid_img[i], dstsize = pyramid_img[i-1].shape[:2])
                lpls = cv2.subtract(pyramid_img[i-1], expend)
                pyramid_lpls.append(lpls)
        return pyramid_lpls
        
    def pyramid_image_fusion(self, img1, img2, fusion_rule, level):
        pyr_gimg1 = self.gaussian_pyramid(img1, level)
        pyr_gimg2 = self.gaussian_pyramid(img2, level)
        pyr_img1 = self.laplacian_pyramid(img1, level)
        pyr_img2 = self.laplacian_pyramid(img2, level)
        pyr_fusion = []
        for i in range(level):
            if fusion_rule == 'weighted':
                temp = self.weighted_average_fusion(pyr_img2[i], pyr_img1[i], 0.7, 0.3)
            elif fusion_rule == 'pca':
                temp = self.PCA_image_fusion(pyr_img2[i], pyr_img1[i])
            else :
                temp = self.HSI_image_fusion(pyr_img2[i], pyr_img1[i])
            pyr_fusion.append(temp)
        ls_ = pyr_gimg1[level-1]
        for i in np.arange(1,level,1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, pyr_fusion[i-1])
        return ls_
       
           
        
           