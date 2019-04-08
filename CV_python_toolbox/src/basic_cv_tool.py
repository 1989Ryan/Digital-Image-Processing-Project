import cv2
import matplotlib.pyplot as plt
import base64
import struct
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from pylab import *
from PIL import Image
from math import *
import cmath
from random import *

class basic_cv_tool:

    def __init__(self, ImageName):
        self.ImageName = ImageName

    def ImageRead(self, ImageName):
        img = cv2.imread(ImageName, cv2.IMREAD_GRAYSCALE)
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
    
    def RGB2HSI(self, rgb_img):
        """
        RGB to HSI
        :param rgm_img: RGB image
        :return: HSI image
        """
        row = np.shape(rgb_img)[0]
        col = np.shape(rgb_img)[1]
        hsi_img = rgb_img.copy()
        B,G,R = cv2.split(rgb_img)
        [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
        H = np.zeros((row, col))    
        I = (R + G + B) / 3.0       
        S = np.zeros((row,col))      
        for i in range(row):
            den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
            thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   
            h = np.zeros(col)               
            h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
            h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
            h[den == 0] = 0
            H[i] = h/(2*np.pi)      
        for i in range(row):
            min = []
            for j in range(col):
                arr = [B[i][j],G[i][j],R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            S[i] = 1 - min*3/(R[i]+B[i]+G[i])
            S[i][R[i]+B[i]+G[i] == 0] = 0
        hsi_img[:,:,0] = H*255
        hsi_img[:,:,1] = S*255
        hsi_img[:,:,2] = I*255
        return hsi_img

    def HSI2RGB(self, hsi_img):
        """
        HSI image transform to RGB image
        :param hsi_img: HSI image
        :return: return RGB image
        """
        # save the shape of original image
        row = np.shape(hsi_img)[0]
        col = np.shape(hsi_img)[1]
        #copy the origin image
        rgb_img = hsi_img.copy()
        #split the channel
        H,S,I = cv2.split(hsi_img)
        #project the channel into [0,1]
        [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
        R,G,B = H,S,I
        for i in range(row):
            h = H[i]*2*np.pi
            #H is bigger than 0 but smaller than 120
            a1 = h >=0
            a2 = h < 2*np.pi/3
            a = a1 & a2         #index in first situation
            tmp = np.cos(np.pi / 3 - h)
            b = I[i] * (1 - S[i])
            r = I[i]*(1+S[i]*np.cos(h)/tmp)
            g = 3*I[i]-r-b
            B[i][a] = b[a]
            R[i][a] = r[a]
            G[i][a] = g[a]
            #H is bigger than 120 but smaller than 240
            a1 = h >= 2*np.pi/3
            a2 = h < 4*np.pi/3
            a = a1 & a2         #index in second situation
            tmp = np.cos(np.pi - h)
            r = I[i] * (1 - S[i])
            g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
            b = 3 * I[i] - r - g
            R[i][a] = r[a]
            G[i][a] = g[a]
            B[i][a] = b[a]
            #H is bigger than 180 but smaller than 360
            a1 = h >= 4 * np.pi / 3
            a2 = h < 2 * np.pi
            a = a1 & a2             
            tmp = np.cos(5 * np.pi / 3 - h)
            g = I[i] * (1-S[i])
            b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
            r = 3 * I[i] - g - b
            B[i][a] = b[a]
            G[i][a] = g[a]
            R[i][a] = r[a]
        rgb_img[:,:,0] = B*255
        rgb_img[:,:,1] = G*255
        rgb_img[:,:,2] = R*255
        return rgb_img

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

    def cdf(self, img):
        hist, bins = np.histogram(img.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf*255/cdf.max()
        cdf = (cdf-cdf[0]) *255/ (cdf[-1]-1)
        cdf = cdf_normalized.astype(np.uint8)
        return cdf
    
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
                    img1[i,j] = 255
        return img1,img2
            
      
    def equalize_histogram(self, img, result_name):
        equ = cv2.equalizeHist(img)
        res = np.hstack((img, equ))
        cv2.imwrite(result_name, res)
        return equ

    def GaussFilter_Kernel_generator(self, sigma, index):
        kernel = np.zeros((index,index), dtype = int8)
        temp = (0-index//2)**2 + (0-index//2)**2
        tr = 1/((1/(2*pi*sigma**2))*np.exp(-temp/2/sigma**2))
        for i in range(index):
            for j in range(index):
                temp = (i-index//2)**2 + (j-index//2)**2
                kernel[i,j]= round(((1/(2*pi*sigma**2))*exp(-temp/2/sigma**2))*tr)
        return kernel/np.sum(kernel)
    
    def GaussFilter(self, img, index):
        kernel = self.GaussFilter_Kernel_generator(1.5, index)
        return signal.convolve2d(img, kernel, boundary='symm', mode='same').astype(int)

    def MediumFilter(self, img, index):
        return cv2.medianBlur(img, index)

    def laplace_filter(self, img):
        kernel = np.matrix([[0,1,0],[1,-4,1],[0,1,0]])
        img = signal.convolve2d(img, kernel, boundary='symm', mode='same').astype(int)
        img[img<0]=0
        return img
     
    def sobel_filter(self, img):
        kernel1 = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])
        kernel2 = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
        img_copy = cv2.copyMakeBorder(img,1,1,1,1, cv2.BORDER_CONSTANT,value=[0,0,0])
        img_new = zeros(np.shape(img),dtype = uint8)
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                img_new[i,j] = abs(np.sum(np.multiply(kernel1, img_copy[i:i+3,j:j+3]))) \
                     + abs(np.sum(np.multiply(kernel2, img_copy[i:i+3,j:j+3])))
        return img_new
    
    def unsharp_mask_filter(self, img, k):
        img_blur = self.GaussFilter(img, 7)
        mask = img - img_blur
        return img + k * mask

    def canny(self, img):
        return cv2.Canny(img.copy(),20,200)
    
    def Transform_matrix(self, d, n, Class, name, img):
        matrix = np.zeros(np.shape(img))
        center = tuple(map(lambda x:(x-1)/2, np.shape(img)))
        for i in range(np.shape(matrix)[0]):
            for j in range(np.shape(matrix)[1]):
                dis = sqrt((center[0] - i) ** 2 + (center[1] - j) ** 2)
                if name == "butterworth":
                       if Class == "low":
                            matrix[i,j] = 1 / (1 + (dis / d) ** (2*n))
                       else :
                            matrix[i,j] = 1 / (1 + (d / dis) ** (2*n))
                elif name == "gauss":
                       if Class == "low":
                            matrix[i,j] = np.exp(-(dis**2)/(2*(d**2)))
                       else :
                            matrix[i,j] = 1 - np.exp(-(dis**2)/(2*(d**2)))
                elif name == "laplace":
                    matrix[i,j] = -4*pi*dis**2/(center[0]**2+center[1]**2)
                else:
                    matrix[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return matrix
    
    def BLPF(self, img, d, n):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        d_matrix = self.Transform_matrix(d, n, "low", "butterworth", img)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        return new_img
        
    
    def GLPF(self, img,d,n):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        d_matrix = self.Transform_matrix(d, n, "low",  "gauss", img)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        return new_img
    
    def BHPF(self, img,d,n):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        d_matrix = self.Transform_matrix(d, n, "high", "butterworth", img)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        return new_img
    
    def GHPF(self, img, d, n):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        d_matrix = self.Transform_matrix(d, n, "high", "gauss", img)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        return new_img
    
    def LHPF(self, img, d, n):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        d_matrix = self.Transform_matrix(d, n, "high", "laplace", img)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        return new_img
    
    def Unsharp_Masking(self, img, d, n, k):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        d_matrix = self.Transform_matrix(d, n, "high", "gauss", img)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        new_img = img + k*new_img
        return new_img
    
    def Power_spectrum_ratio(self, img, img_new):
        #center = tuple(map(lambda x:(x-1)/2, np.shape(img)))
        p = np.zeros(np.shape(img))
        p1 = np.zeros(np.shape(img))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        f2 = np.fft.fft2(img_new)
        fshift2 = np.fft.fftshift(f2)
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                p[i,j] = abs(fshift[i,j])**2
                p1[i,j] = abs(fshift2[i,j])**2
        return np.sum(p1)/np.sum(p)    

    def Gaussian_Noise_generator(self, img, sigma, mu, k):
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                tmprand = gauss(mu, sigma)
                if tmprand<0 or tmprand>255:
                    continue
                img[i,j] = img[i,j] + k * tmprand
                if img[i,j] > 255:
                    img[i,j] = 255
                elif img[i,j]<0:
                    img[i,j] = 0
        return img
    
    def Salt_Noise_generator(self, img, perc):
        num = int(np.size(img)*perc)
        for i in range(num):
            x = randint(0, np.shape(img)[0] - 1)
            y = randint(0, np.shape(img)[1] - 1) 
            img[x][y] = 255
        return img
    
    def Pepper_Noise_generator(self, img, perc):
        num = int(np.size(img)*perc)
        for i in range(num):
            x = randint(0, np.shape(img)[0] - 1)
            y = randint(0, np.shape(img)[1] - 1) 
            img[x][y] = 0
        return img


    def Salt_and_Pepper_Noise_generator(self, img, perc):
        num = int(np.size(img)*perc)
        for i in range(num):
            x = randint(0, np.shape(img)[0] - 1)
            y = randint(0, np.shape(img)[1] - 1)
            if randint(0,1) == 0: 
                img[x][y] = 255
            else:
                img[x][y] = 0
        return img

    def meanFilter(self, img, index1, index2):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = np.mean(img_copy[i:i+index1,j:j+index2])
                img_temp[i,j] = temp
        return img_temp

    def geo_meanFilter(self, img, index1, index2):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = 1.0
                num = 0
                for m in range(index1):
                    for n in range(index2):
                        if img_copy[i+m, j+n]!=0:
                            temp = temp*img_copy[i+m, j+n] 
                            num += 1
                img_temp[i,j] = pow(temp,1/num)
        return img_temp
    
    def har_meanFilter(self, img, index1, index2):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = 0.0
                for m in range(index1):
                    for n in range(index2):
                        if img_copy[i+m, j+n]!=0:
                            temp = temp+1/img_copy[i+m, j+n] 
                img_temp[i,j] = index1*index2/temp
        return img_temp

    def contraharm_meanFilter(self, img, index1, index2,q):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp1 = 0.0
                temp2 = 0.0
                for m in range(index1):
                    for n in range(index2):
                        temp1 = temp1 + img_copy[i+m, j+n]**(q+1)
                        temp2 = temp2 + img_copy[i+m, j+n]**q
                img_temp[i,j] = temp1/temp2
        return img_temp

    def maxFilter(self, img, index1, index2):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = np.max(img_copy[i:i+index1,j:j+index2])
                img_temp[i,j] = temp
        return img_temp
    
    def minFilter(self, img, index1, index2):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = np.min(img_copy[i:i+index1,j:j+index2])
                img_temp[i,j] = temp
        return img_temp
    
    def midFilter(self, img, index1, index2):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = np.max(img_copy[i:i+index1,j:j+index2])/2+np.min(img_copy[i:i+index1,j:j+index2])/2
                img_temp[i,j] = temp
        return img_temp
    
    def fixed_alpha_meanFilter(self, img, index1, index2, d):
        img_temp = img.copy()
        img_copy = cv2.copyMakeBorder(img,(index1-1)//2,(index1-1)//2,(index2-1)//2,(index2-1)//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        for i in range(np.shape(img)[0]):
            for j in range(np.shape(img)[1]):
                temp = np.array(img_copy[i:i+index1,j:j+index2])
                temp = temp.flatten()
                temp = np.sort(temp)
                temp = np.sum(temp[int(d/2):index1*index2-int(d/2)])
                img_temp[i,j] = temp/(index1*index2-d)
        return img_temp

    def Blur_filter(self, img, T, a, b):
        Img = np.fft.fft2(img)
        Img = np.fft.fftshift(Img)
        shape = np.shape(Img)
        H = np.zeros((shape[0], shape[1]), dtype = complex)
        G = np.zeros((shape[0], shape[1]), dtype = complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                phase = pi*((i+1)-shape[0]/2)*a + pi*((j+1)-shape[0]/2)*b
                if phase == 0:
                    H[i,j] = T
                else:
                    H[i,j] = T/phase*sin(phase)*e**(-cmath.sqrt(-1)*phase)
                G[i,j] = H[i,j]*Img[i,j]
        return G, H

    def Wenier_filter(self, H, K, G):
        shape = np.shape(G)
        F = np.zeros(shape, dtype = complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                F[i,j] = 1/H[i,j]*(np.abs(H[i,j])**2/(np.abs(H[i,j])**2+K))*G[i,j]
        return F

    def CLSF(self, H, G, r):
        shape = np.shape(G)
        p = np.matrix([[0,-1,0],[-1,4,-1],[0,-1,0]])
        p = np.pad(p, (shape[0]/2-2, shape[0]/2), 'constant')
        P = np.fft.fft2(p)
        P = np.fft.fftshift(P)
        F = np.zeros(shape, dtype = complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                F[i,j] = H[i,j].conjugate()/(np.abs(H[i,j])**2+r*np.abs(P[i,j])**2)*G[i,j]
        return F
