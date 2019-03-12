import cv2
import matplotlib.pyplot as plt
import base64
import struct
import numpy as np
from scipy.interpolate import interp1d
from pylab import *
from PIL import Image

class basic_cv_tool:

    def __init__(self, ImageName):
        self.ImageName = ImageName

    def ImageRead(self, ImageName):
        img = cv2.imread(ImageName, 0)
        return img
    
    def BMP_information_analysis(self, ImageName):
        with open(ImageName, 'rb') as f:
            raw_info = f.read(30)
        info = struct.unpack('<ccIIIIIIHH', raw_info)
        if(info[0]!=b'B' or info[1] !=b'M'):
            return None
        else:
            return {
                'size' : info[2],
                'bias' : info[4],
                'header' : info[5],
                'width' : info[6], 
                'height' : info[7],
                'color_bit' : info[9]
            }

    def greyscale_reduce(self, img, reduce_index):
        shape = img.shape
        width = shape[0]
        height = shape[1]
        for i in range(width):
            for j in range(height):
                for k in range(3):
                    img[i,j,k] =(img[i,j,k]/ reduce_index) *(255 /(255 / reduce_index))
        return img

    def image_average(self, img):
        mean = np.mean(img)
        return mean
    
    def image_variance(self, img):
        var = np.var(img)
        return var

    def image_Nearest_neighbor_interpolation(self, img, Zoom_index):
        img = cv2.resize(img, Zoom_index, interpolation = cv2.INTER_NEAREST)

        return img
    
    def image_bilinear_interpolation(self, img, Zoom_index):
        img = cv2.resize(img, Zoom_index, interpolation = cv2.INTER_LINEAR)
        return img
    
    def image_bicubic_interpolation(self, img, Zoom_index):
        img = cv2.resize(img, Zoom_index, interpolation = cv2.INTER_CUBIC)
        return img

    def image_shear(self, img, shear_index):
        shear_matrix =np.array([
            [1,shear_index,0],
            [0,1,0]
            ],dtype=np.float32)
        img = cv2.warpAffine(img, shear_matrix, (int(img.shape[0]*(1+shear_index)),img.shape[1]))
        return img
    
    def image_rotation(self, img, rotation_theta):
        theta=rotation_theta*np.pi/180
        rotate_matrix=np.array([
            [np.cos(theta),-np.sin(theta),np.sin(theta)*img.shape[0]],
            [np.sin(theta),np.cos(theta),0]
            ],dtype=np.float32)
        img = cv2.warpAffine(img,rotate_matrix, (int(img.shape[0]*(np.cos(theta)+np.sin(theta))),int(img.shape[1]*(np.cos(theta)+np.sin(theta)))))
        return img
    
    def interest_point_choosing(self, ImageName):
        img = array(Image.open(ImageName))
        imshow(img)
        fea_point = ginput(7)
        fea_point = np.float32(fea_point)
        fea_point = np.column_stack((fea_point,array([1,1,1,1,1,1,1])))
        return fea_point
    
    def Getting_H_Matrix(self, img_points_1, img_points_2):
        H_matrix = ((img_points_2.transpose()).dot(img_points_1)).dot(np.linalg.inv((img_points_1.transpose()).dot(img_points_1)))
        print(H_matrix)
        return H_matrix[:2]

    def calcdf(self, img):
        hist, bins = np.histogram(img.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf*255/cdf.max()
        cdf = (cdf-cdf[0]) *255/ (cdf[-1]-1)
        cdf = cdf_normalized.astype(np.uint8)
        temp = np.zeros(256,dtype = np.uint8)
        j = 0
        for i in range(256):
            j = cdf[i]
            temp[j]=i
        for i in range(255):
            if temp[i+1]<temp[i]:
                temp[i+1] = temp[i]
        return temp
    
    def createcdf(self, hist):
        cdf = hist.cumsum()
        cdf_normalized = cdf*255/cdf.max()
        cdf = (cdf-cdf[0]) *255/ (cdf[-1]-1)
        cdf = cdf_normalized.astype(np.uint8)
        temp = np.zeros(256,dtype = np.uint8)
        j = 0
        for i in range(256):
            j = cdf[i]
            temp[j]=i
        for i in range(255):
            if temp[i+1]<temp[i]:
                temp[i+1] = temp[i]
        return temp
    
    def createhisto(self, array):
        array = np.array(array)
        x = np.linspace(0,255,shape(array)[0])
        f = interp1d(x,array,kind='linear')
        x_pred=np.linspace(0,255,256)
        arr = f(x_pred)
        return arr
    
    def histo_matching(self, img, cdf):
        res = np.zeros((512, 512, 3), dtype =np.uint8)
        res = cdf[img]
        return res
    
    def local_histo(self, img, index):
        img_copy = cv2.copyMakeBorder(img,(index-1)//2,(index-1)//2,(index-1)//2,(index-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = cv2.equalizeHist(img_copy[i:i+index,j:j+index])
                img[i,j] = temp[(index-1)//2,(index-1)//2]
        return img
    
    def segmentation(self, img):
        T = 30
        color = np.linspace(0,255,256)
        hist , bins = np.histogram(img.flatten(), 256,[0,256])
        print("image mean value is", T)
        while(1):
            T1 = (hist[:T]*color[:T]).sum()/hist[:T].sum()
            T2 = (hist[T:]*color[T:]).sum()/hist[T:].sum()
            temp = T
            T = int((T1+T2)/2)
            print("T1",T1,"T2",T2,"T",T)
            if abs(temp-T)<0.01:
                break
        img1 = zeros((shape(img)[0],shape(img)[1],3),dtype = uint8)
        img2 = zeros((shape(img)[0],shape(img)[1],3),dtype = uint8)
        for i in range(shape(img)[0]):
            for j in range(shape(img)[1]):
                if img[i,j]<T:
                    img1[i,j] = img[i,j]
                else:
                    img2[i,j] = img[i,j]
        return img1,img2
            
      
    def equalize_histogram(self, img, result_name):
        equ = cv2.equalizeHist(img)
        res = np.hstack((img, equ))
        cv2.imwrite(result_name, res)
        return equ
    
      