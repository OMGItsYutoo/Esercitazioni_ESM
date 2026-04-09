import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def block_fun(x,T):
    k=int(np.sqrt(x.shape[0]))
    x=np.reshape(x,(k,k))
    
    x_c=(k-1)//2
    y_c=x_c
    
    sectAv=x[:,:x_c]
    sectBv=x[:,x_c+1:]
    va=np.mean(sectAv)
    vb=np.mean(sectBv)
    
    rv=np.max([va/vb,vb/va])
    
    sectAh=x[:y_c,:]
    sectBh=x[y_c+1:,:]
    va=np.mean(sectAh)
    vb=np.mean(sectBh)
    
    rh=max(va/vb,vb/va)

    r=max(rv,rh)
    
    if r>T:
        return 1
    else: 
        return 0

def ratio_detector(x,k,T):
    return ndi.generic_filter(x,block_fun,(k,k),extra_arguments=(T,))

def main():
    plt.close('all')
    
    x=np.fromfile("images/target_rumorosa.raw",dtype=np.float32)
    x=np.float64(np.reshape(x,(256,256)))
    
    k=3;T=1.6
    map_bor=ratio_detector(x,k,T)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(map_bor,clim=[0,1],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()