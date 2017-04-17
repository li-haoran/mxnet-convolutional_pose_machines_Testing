import matplotlib.pyplot as plt
import numpy as np
import cv2
from pylab import quiver,quiverkey
def heat_plot(image,map,num):
    '''
    image: input image
    map:list of heat map
    num: map
    '''
    
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(6, 6))
    axe=axes.ravel()
    (height,width,depth)=image.shape
    im1 = axe[15].imshow(image, extent=(0,width,0,height))
    for i in range(num):
        im1 = axe[i].imshow(image, extent=(0,width,0,height))
        axe[i].hold(True)
        axe[i].axis('off')
        im2 = axe[i].imshow(map[:,:,i], cmap=plt.cm.jet, alpha=0.75, interpolation='bilinear',extent=(0,width,0,height))
    im1 = axe[15].imshow(image, extent=(0,width,0,height))
    plt.show()
    #fig.savefig('test.png', dpi=None, facecolor='w', edgecolor='w',
    #        orientation='portrait', papertype=None, format=None,
    #        transparent=False, bbox_inches=None, pad_inches=0.1,
    #        frameon=None)
    
    print 'done'