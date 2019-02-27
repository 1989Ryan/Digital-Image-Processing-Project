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

image_1_name = '../../homework1/7.bmp'

tool = basic_cv_tool(image_1_name)
info = tool.BMP_information_analysis(image_1_name)
print('size: ', info['size'], ', bias: ', info['bias'], ', header: ', info['header'], ', width: ', info['width'], \
' height: ', info['height'], ', color: ', info['color_bit'])

