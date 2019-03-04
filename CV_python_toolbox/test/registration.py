import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.basic_cv_tool import *

'''
This is the test file for project No.1 which consists of all the required 
assignments.
'''

image_1_name = '../../homework2/Image A.jpg'
image_2_name = '../../homework2/Image B.jpg'

tool = basic_cv_tool(image_1_name)
image = tool.ImageRead(image_1_name)
print(image.shape)
img1 = tool.interest_point_choosing(image_1_name)
img2 = tool.interest_point_choosing(image_2_name)
M = tool.Getting_H_Matrix(img1, img2)
print(M)
img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
print(img.shape)
cv2.imwrite('test.jpg', img)