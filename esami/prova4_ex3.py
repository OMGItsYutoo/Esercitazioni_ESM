import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def main():
    plt.close('all')
   
    x1=np.float64(io.imread("images/img1.png"))
    x2=np.float64(io.imread("images/img2.png"))

    y1=ndi.gaussian_filter(x1,10)<20
    y2=ndi.gaussian_filter(x2,10)<20
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(x1,clim=[0,255],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(x2,clim=[0,255],cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(y1,clim=[0,1],cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(y2,clim=[0,1],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()

