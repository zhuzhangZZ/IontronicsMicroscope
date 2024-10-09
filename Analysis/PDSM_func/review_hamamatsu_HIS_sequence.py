"""
script for previewing HIS sequences based
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from readHISbasic import readSection #based on readHISbasic version 2018/07/13

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        #axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


filedir = 'D:/FiberTip/180627_Channel_FLbeads/'
fn = filedir + '180627FL500nm_Chnl5um_Exp10ms00002.HIS'
figures = {}  #empty dictionary for adding figures
m = np.memmap(fn, shape=None, mode = 'r')
offset = 0
for i in range(9):
    try:
        img = readSection(m, offset)
    except:
        break
    offset = img.HIS.offsetNext
    figures['im'+str(i)] = np.copy(img)
    # print(i, time.time()*1000) to check loading time

# plot of the images in a figure, with 5 rows and 4 columns

plot_figures(figures, 3, 3)
plt.show()



