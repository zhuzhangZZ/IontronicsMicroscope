"""
Collection of functions for extracting information from the images of particle at the end of a round capillary

__author__  = "Sanli Faez <s.faez@uu.nl>"
__license__ = "BSD license"

first line: 13 July 2018



current version: 20180713
"""

import numpy as np
import matplotlib.pyplot as plt



def averageIntensity(image, filters):
    """
    image: numpy array
    filters: set of mask filters

    output: array of average values for each mask in filters
    """
    intensities = np.zeros([1,len(filters)])
    i=0
    for f in filters:
        area = filters[f]
        intensities[0, i] = np.sum(area*image)/np.sum(area)
        i = i+1

    return intensities

def averageIntensityMasks(image, c, r, dr):
    """
    image: numpy array
    c: coordinate of the center of symmetry
    r: array of radiai of the lower limits of strips, must be >=0
    dr: radial width of the strip that has to be averaged

    output: array of mask filter of the same size of image to be overlayed with the image for averaging
    """
    # creating the area of interest
    [m, n] = np.shape(image)
    filters = {}  #dictionary of filters correspoding to stripes of certain radius
    for rad in r:
        interest = image * 0
        for i in range(m):
            for j in range(n):
                rij = np.sqrt((i-c[0])**2 + (j-c[1])**2)
                if rad <= rij and rij < (rad+dr):
                    interest[i,j] = 1

        filters['im'+str(rad)] = np.copy(interest)

    return filters

####usage test
# test = np.random.rand(20,30)
# f = averageIntensityMasks(test, [10,10], [0,3,6], 2)
# for i in f:
#     plt.imshow(f[i])
#     plt.show()


