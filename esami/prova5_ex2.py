import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def fast_std(x,k):
    return np.sqrt(ndi.uniform_filter(x**2,(k,k))-ndi.uniform_filter(x,(k,k))**2)

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/cells.png"))

    x=x[:,:,0]
    
    mask=x>51
    
    h=np.array([[1,1,1],
                [1,-8,1],
                [1,1,1]])

    y=ndi.correlate(mask,h)
    
    #y=fast_std(x,7)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(mask,clim=[0,1],cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(y,clim=[0,1],cmap='gray')
    
    low=51
    high=150   
    xp1=ndi.uniform_filter(x,(15,15))<high
    xp2=ndi.uniform_filter(xp1,(15,15))>(high/255)
    xp3=x*xp2
    y=xp3>low

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(xp2,clim=[0,1],cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(y,cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()