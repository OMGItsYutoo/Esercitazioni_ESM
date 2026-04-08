import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def block_fun(x):
    y=(x-np.mean(x))**2
    return np.sqrt(np.sum(y))

def detect(p1,p2,k):
    p1_mean=ndi.uniform_filter(p1,(k,k))
    p2_mean=ndi.uniform_filter(p2,(k,k))
    
    num=ndi.uniform_filter(p1*p2,(k,k))-p1_mean*p2_mean
    num*=127*127
    
    den1=ndi.generic_filter(p1,block_fun,(k,k))
    den2=ndi.generic_filter(p2,block_fun,(k,k))
    
    mappa=num/(den1*den2)
    
    mask = mappa<0.02
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mappa, cmap='jet', clim=None)
    plt.colorbar()
    plt.title('Correlazione')
    plt.subplot(1,2,2)
    plt.imshow(mask, clim=(0,1), cmap='gray')
    plt.title('Maschere')
    return mask
    
def main():
    plt.close('all')
    
    p1=np.load("images/data_P1.npy")
    p2=np.load("images/data_P2.npy")
    img=np.load("images/data_img.npy")

    detect(p1,p2,127)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(p1,clim=[0,1],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(p2,clim=[0,1],cmap='gray')

    plt.figure()
    plt.imshow(img,clim=[0,255],cmap='gray')

if __name__=='__main__':
    main()
    plt.show()

