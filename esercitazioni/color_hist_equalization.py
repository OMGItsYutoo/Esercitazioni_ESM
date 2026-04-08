import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import histogram, equalize_hist


def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/volto.tiff'))/255
    
    plt.figure()
    plt.imshow(x)
    
    x_hsv=rgb2hsv(x)
    
    V=x_hsv[:,:,2]
    
    V_new=equalize_hist(V)
    
    plt.subplot(1,3,4)
    plt.imshow(C, clim=[0,1], cmap='gray')
    plt.subplot(1,3,5)
    plt.imshow(M, clim=[0,1], cmap='gray')
    plt.subplot(1,3,6)
    plt.imshow(Y, clim=[0,1], cmap='gray')

if __name__=='__main__':
    main()
    
    plt.show()
    