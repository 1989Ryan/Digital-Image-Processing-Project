import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.basic_cv_tool import *

'''This is the test file for project No.3 which consists of all the required 
assignments.
'''
def equalized_histogram(imagename):
    image_name1 = "../../homework3/project3/"+imagename+".bmp"
    image_cdf1 = "../../homework3/project3/"+imagename+"_cdf.png"
    result_name1 = "../../homework3/result_"+imagename+".bmp"
    result_cdf1 = "../../homework3/result_"+imagename+"_cdf.png"
    result_hist = "../../homework3/result_"+imagename+"_hist.png"
    tool = basic_cv_tool(image_name1)
    img = tool.ImageRead(image_name1)
    cdf = tool.calcdf(img)
    plt.plot(cdf, color = 'b')
    plt.savefig(image_cdf1)
    equ = tool.equalize_histogram(img, result_name1)
    plt.close()
    plt.figure()
    cdf = tool.calcdf(equ)
    plt.plot(cdf, color = 'b')
    plt.savefig(result_cdf1)
    plt.close()
    plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    p1.set_title('original histogram',fontsize = 11)
    p1.hist(img.ravel(),256,[0,255])
    p2.set_title('equalized histogram', fontsize = 11)
    p2.hist(equ.ravel(),256,[0,255])
    plt.savefig(result_hist)
    plt.close()

def histogram_specialization(imagename, cdf):
    image_name = "../../homework3/project3/"+imagename+".bmp"
    result_name = "../../homework3/mat_result_"+imagename+".bmp"
    result_hist = "../../homework3/mat_result_"+imagename+"_hist.png"
    tool = basic_cv_tool(image_name)
    img = tool.ImageRead(image_name)
    equ = cv2.equalizeHist(img)
    mat = tool.histo_matching(equ, cdf)
    res = np.hstack((img,equ, mat))
    cv2.imwrite(result_name, res)
    plt.figure(figsize=(7,9),dpi=98)
    p1 = plt.subplot(311)
    p2 = plt.subplot(312)
    p3 = plt.subplot(313)
    p1.hist(img.ravel(),256,[0,255])
    p1.set_title('original histogram',fontsize = 11)
    p2.hist(equ.ravel(),256,[0,255])
    p2.set_title('equalized histogram',fontsize= 11)
    p3.hist(mat.ravel(),256,[0,255])
    p3.set_title('specialized histogram',fontsize = 11)
    plt.savefig(result_hist)
    plt.close()
    '''
    mat = tool.histo_matching(img, cdf)
    res = np.hstack((img, mat))
    cv2.imwrite(result_name, res)
    plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    p1.hist(img.ravel(),256,[0,255])
    p1.set_title('original histogram',fontsize = 11)
    p2.hist(mat.ravel(),256,[0,255])
    p2.set_title('specialized histogram',fontsize = 11)
    plt.savefig(result_hist)
    plt.close()
    '''

def local_histogram(imagename,index):
    image_name = "../../homework3/project3/"+imagename+".bmp"
    result_name = "../../homework3/local_result_"+imagename+".bmp"
    result_hist = "../../homework3/local_result_"+imagename+"_hist.png"
    tool = basic_cv_tool(image_name)
    img = tool.ImageRead(image_name)
    plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    p1.hist(img.ravel(),256,[0,255])
    p1.set_title('original histogram',fontsize = 11)
    loc = tool.local_histo(img, index)
    cv2.imwrite(result_name, loc)
    p2.hist(loc.ravel(),256,[0,255])
    p2.set_title('local equalized histogram',fontsize = 11)
    plt.savefig(result_hist)
    plt.close()
    

def calcdf(imagename):
    img_name = "../../homework3/project3/"+imagename+".bmp"
    tool = basic_cv_tool(img_name)
    img = tool.ImageRead(img_name)
    cdf = tool.calcdf(img)
    return cdf

def hist_segmentation(imagename):
    image_name = "../../homework3/project3/"+imagename+".bmp"
    result_name1 = "../../homework3/seg_result1_"+imagename+".bmp"
    result_name2 = "../../homework3/seg_result2_"+imagename+".bmp"
    result_hist = "../../homework3/seg_result_"+imagename+"_hist.png"
    tool = basic_cv_tool(image_name)
    img = tool.ImageRead(image_name)
    img1, img2 = tool.segmentation(img)
    plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    p1.hist(img1.ravel(),256,[0,255])
    p1.set_title('image1 histogram',fontsize = 11)
    #loc = tool.local_histo(img1, index)
    cv2.imwrite(result_name1, img1)
    cv2.imwrite(result_name2, img2)
    p2.hist(img2.ravel(),256,[0,255])
    p2.set_title('image2 histogram',fontsize = 11)
    plt.savefig(result_hist)
    plt.close()



if __name__ == '__main__':
    '''
    Assignment 1, equalized histogram transformation
    equalized_histogram('lena')
    equalized_histogram('elain')
    equalized_histogram('lena1')
    equalized_histogram('lena2')
    equalized_histogram('lena4')
    equalized_histogram('elain1')
    equalized_histogram('elain2')
    equalized_histogram('elain3')
    equalized_histogram('citywall')
    equalized_histogram('citywall1')
    equalized_histogram('citywall2')
    Assignment 2, specialized histogram transformation
    cdf1 = calcdf('lena')
    cdf2 = calcdf('elain')
    cdf3 = calcdf('citywall')
    histogram_specialization('lena1',cdf1)
    histogram_specialization('lena2',cdf1)
    histogram_specialization('lena4',cdf1)
    histogram_specialization('elain1',cdf2)
    histogram_specialization('elain2',cdf2)
    histogram_specialization('elain3',cdf2)
    histogram_specialization('citywall1',cdf3)
    histogram_specialization('citywall2',cdf3)
    '''
    '''
    Assignment 3, local histogram transformation using equalization transformation.
    '''
    '''
    local_histogram('lena',7)
    local_histogram('elain',7)
    '''
    hist_segmentation('elain')
    