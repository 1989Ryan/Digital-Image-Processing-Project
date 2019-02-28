import cv2
import matplotlib.pyplot as plt
import base64
import struct
import numpy as np

from PIL import Image

class basic_cv_tool:

    def __init__(self, ImageName):
        self.ImageName = ImageName

    def ImageRead(self, ImageName):
        img = cv2.imread(ImageName)
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

    def image_Nearest_neighbor_interpolation(self, ImageName, Zoom_index):
        #TODO
        pass
    
    def image_bilinear_interpolation(self, ImageName, Zoom_index):
        #TODO
        pass
    
    def image_bicubic_interpolation(self, ImageName, Zoom_index):
        #TODO
        pass

    def image_shear(self, ImageName, shear_index):
        #TODO
        pass
    
    def image_rotation(self, ImageName, rotation_degree):
        #TODO
        pass