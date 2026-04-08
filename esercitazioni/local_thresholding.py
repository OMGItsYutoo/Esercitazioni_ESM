import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/yeast.tif"))
    
    k=3
    mean_of_square=ndi.uniform_filter(x**2,(k,k))
    square_of_mean=ndi.uniform_filter(x,(k,k))**2
    std=np.sqrt(mean_of_square-square_of_mean)
    
    a=30;b=1.5
    mask=(x>(a*std)) & (x>b*np.mean(x))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask,clim=[0,1],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()