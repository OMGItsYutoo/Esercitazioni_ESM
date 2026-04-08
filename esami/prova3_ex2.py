import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

def fast_var(x,k):
    return ndi.uniform_filter(x**2,(k,k))-ndi.uniform_filter(x,(k,k))**2

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/ala_ape.jpg"))/255
    
    x_hsv=rgb2hsv(x)
    
    H=x_hsv[:,:,0]
    S=x_hsv[:,:,1]
    V=x_hsv[:,:,2]
    
    mask_at=ndi.gaussian_filter(S,1)>0.25
    L=np.array([[1,1,1],[1,-8,1],[1,1,1]])    
    lap_at= ndi.correlate(mask_at,L)**2
    
    mask=fast_var(S,3)
    mask=mask>0.02
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,cmap='gray',clim=[0,1])
    plt.subplot(1,3,2)
    plt.imshow(mask,cmap='gray',clim=[0,1])
    plt.subplot(1,3,3)
    plt.imshow(lap_at,cmap='gray',clim=[0,1])
    
if __name__=='__main__':
    main()
    plt.show()



