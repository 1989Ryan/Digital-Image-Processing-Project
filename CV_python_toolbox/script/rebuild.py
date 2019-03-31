import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.basic_cv_tool import *

image_name = '../../homework6/project6/lena.bmp'

tool = basic_cv_tool(image_name)
img = tool.ImageRead(image_name)
img1 = tool.Gaussian_Noise_generator(img.copy(), 20, 20, 1)
img2 = tool.Salt_and_Pepper_Noise_generator(img.copy(), 0.1)
img3 = tool.geo_meanFilter(img1.copy(),3,3)
img4 = tool.geo_meanFilter(img2.copy(),3,3)
cv2.imwrite("../../homework6/gausslena.bmp",img1)
cv2.imwrite("../../homework6/saltpepperlena.bmp",img2)
cv2.imwrite("../../homework6/meangausslena.bmp",img3)
cv2.imwrite("../../homework6/meansaltpepperlena.bmp",img4)