import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/space.jpg"))
    
    k=5
    y=ndi.uniform_filter(x,(k,k))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255],cmap='gray')
    
    max=np.max(y)
    y=y>(0.25*max)
    x_new=x*y
    
    plt.figure()
    plt.imshow(x_new, clim=[0,255],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()

