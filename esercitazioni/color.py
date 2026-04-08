import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/cubo.jpg'))/255
    
    plt.figure()
    plt.imshow(x)
    
    R=x[:,:,0]
    G=x[:,:,1]
    B=x[:,:,2]
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(R, clim=[0,1], cmap='gray')
    plt.subplot(2,3,2)
    plt.imshow(G, clim=[0,1], cmap='gray')
    plt.subplot(2,3,3)
    plt.imshow(B, clim=[0,1], cmap='gray')

    C=1-R
    M=1-G
    Y=1-B    
    
    plt.subplot(2,3,4)
    plt.imshow(C, clim=[0,1], cmap='gray')
    plt.subplot(2,3,5)
    plt.imshow(M, clim=[0,1], cmap='gray')
    plt.subplot(2,3,6)
    plt.imshow(Y, clim=[0,1], cmap='gray')

    z=np.stack((C,M,Y),2)
    K=np.min(z,2)    
    
    #K=np.minimum(np.minimum(C,M),Y)
    
    Cn=C-K
    Mn=M-K
    Yn=Y-K
    
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(Cn, clim=[0,1], cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(Mn, clim=[0,1], cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(Yn, clim=[0,1], cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow(K, clim=[0,1], cmap='gray')
    
    x_hsv=rgb2hsv(x)
    
    H=x_hsv[:,:,0]
    S=x_hsv[:,:,1]
    V=x_hsv[:,:,2]
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(H, clim=[0,1], cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(S, clim=[0,1], cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(V, clim=[0,1], cmap='gray')

if __name__=='__main__':
    main()
    
    plt.show()
    