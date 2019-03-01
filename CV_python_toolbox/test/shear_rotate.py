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
new_img = tool.image_shear(img, 1.5)
'''
new_img1 = tool.image_bicubic_interpolation(new_img,(2048,2048))
new_img2 = tool.image_bilinear_interpolation(new_img,(2048,2048))
new_img3 = tool.image_Nearest_neighbor_interpolation(new_img,(2048,2048))

new_img1 = Image.fromarray(new_img1)
new_img2 = Image.fromarray(new_img2)
new_img3 = Image.fromarray(new_img3)
new_img1.save('elainshearcubic.bmp')
new_img1.save('elainshearcubic.png')
new_img2.save('elainshearlinear.bmp')
new_img2.save('elainshearlinear.png')
new_img3.save('elainshearnear.bmp')
new_img3.save('elainshearnear.png')
'''
new_img = Image.fromarray(new_img)
new_img.save('lenashearcubic.png')