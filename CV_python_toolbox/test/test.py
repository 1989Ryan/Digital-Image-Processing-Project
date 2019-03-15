import cv2
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gaussian(size, u = 0, sigma = 0.85):
	filters = np.zeros((size[0], size[1]))
	k = int(size[0] / 2)
	for i in range(size[0]):
		for j in range(size[1]):
			filters[i, j] = g(u, sigma, i, j, k)
	evid = 1.0 / filters[0, 0]
	for i in range(size[0]):
		for j in range(size[1]):
			filters[i, j] = int(filters[i, j] * evid)
	return filters 

def g(u, sigma, i, j, k):
	weight = math.exp(((-(i-k)**2-(j-k)**2)/2/sigma**2))/2/math.pi/sigma**2
	return weight

if __name__ == '__main__':
    print(gaussian([3,3]))