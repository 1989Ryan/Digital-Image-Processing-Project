import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.basic_cv_tool import *

image_2_name = '../../homework1/lena.bmp'

tool = basic_cv_tool(image_2_name)
img = tool.ImageRead(image_2_name)
mean = tool.image_average(img)
var = tool.image_variance(img)
print(mean)
print(var)