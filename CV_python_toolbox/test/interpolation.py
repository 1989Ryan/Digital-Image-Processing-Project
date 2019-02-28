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

image_2_name = '../../homework1/lena.bmp'

tool = basic_cv_tool(image_2_name)
img = tool.ImageRead(image_2_name)
new_img = tool.image_bicubic_interpolation(img,(2048,2048))
new_img = Image.fromarray(new_img)
new_img.show()