import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.basic_cv_tool import *


'''This is the script for project No.4 which consists of all the required 
assignments.
'''

def gauss_process(imagename):
    image_name1 = "../../homework4/project4/"+imagename
    result = "../../homework4/result1"+imagename[:5]+".png"
    result2 = "../../homework4/result1"+imagename[:5]+"1.png"
    tool = basic_cv_tool(image_name1)
    img = tool.ImageRead(image_name1)
    img1 = tool.GaussFilter(img, 3)
    img2 = tool.GaussFilter(img, 5)
    img3 = tool.GaussFilter(img, 7)
    plt.figure(figsize = (16,5))
    p1 = plt.subplot(131)
    p1.set_title('gaussfilter, 3x3', fontsize = 11)
    p1.imshow(img1,cmap='gray')
    p2 = plt.subplot(132)
    p2.set_title('gaussfilter, 5x5', fontsize = 11)
    p2.imshow(img2,cmap='gray')
    p3 = plt.subplot(133)
    p3.set_title('gaussfilter, 7x7', fontsize = 11)
    p3.imshow(img3,cmap='gray')
    plt.savefig(result)
    res = np.hstack((img1, img2, img3))
    cv2.imwrite(result2,res)
    

def mid(imagename):
    image_name1 = "../../homework4/project4/"+imagename
    result = "../../homework4/result2"+imagename[:5]+".png"
    result2 = "../../homework4/result2"+imagename[:5]+"1.png"
    tool = basic_cv_tool(image_name1)
    img = tool.ImageRead(image_name1)
    img1 = tool.MediumFilter(img, 3)
    img2 = tool.MediumFilter(img, 5)
    img3 = tool.MediumFilter(img, 7)
    plt.figure(figsize = (16,5))
    p1 = plt.subplot(131)
    p1.set_title('midfilter, 3x3', fontsize = 11)
    p1.imshow(img1,cmap='gray')
    p2 = plt.subplot(132)
    p2.set_title('midfilter, 5x5', fontsize = 11)
    p2.imshow(img2,cmap='gray')
    p3 = plt.subplot(133)
    p3.set_title('midfilter, 7x7', fontsize = 11)
    p3.imshow(img3,cmap='gray')
    plt.savefig(result)
    res = np.hstack((img1, img2, img3))
    cv2.imwrite(result2,res)


def high_pass_filter_process(imagename):
    image_name1 = "../../homework4/project4/"+imagename
    result = "../../homework4/result2"+imagename[:5]+".png"
    result2 = "../../homework4/result3"+imagename[:5]+"1.png"
    tool = basic_cv_tool(image_name1)
    img = tool.ImageRead(image_name1)
    img1 = tool.laplace_filter(img)
    img2 = tool.sobel_filter(img)
    img3 = tool.unsharp_mask_filter(img, 0.5)
    img4 = tool.canny(img)
    plt.figure(figsize = (16,4))
    p1 = plt.subplot(141)
    p1.set_title('laplace', fontsize = 11)
    p1.imshow(img1,cmap='gray')
    p2 = plt.subplot(142)
    p2.set_title('sobel', fontsize = 11)
    p2.imshow(img2,cmap='gray')
    p3 = plt.subplot(143)
    p3.set_title('unsharp', fontsize = 11)
    p3.imshow(img3,cmap='gray')
    p3 = plt.subplot(144)
    p3.set_title('canny', fontsize = 11)
    p3.imshow(img4,cmap='gray')
    plt.savefig(result)
    res = np.hstack((img1, img2, img3, img4))
    cv2.imwrite(result2,res)

if __name__ == '__main__':
    gauss_process("test1.pgm")
    gauss_process("test2.tif")
    mid("test1.pgm")
    mid("test2.tif")
    high_pass_filter_process("test3_corrupt.pgm")
    high_pass_filter_process("test4.tif")