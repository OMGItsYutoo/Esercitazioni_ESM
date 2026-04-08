import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

def block_fun(x):
    rag=np.mean(x**2)/(np.prod(x**2))**(1/(len(x)))
    return rag
    
def edge_detection(x,k,T):
    rag=ndi.generic_filter(x,block_fun,(k,k))
    return rag>=T

def main():
    plt.close('all')
    
    x=np.fromfile("images/target_rumorosa.raw",dtype=np.float32)
    x=np.reshape(x,(256,256))

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,cmap='gray',clim=[0,255])    
    
    k=3
    x_smooth=ndi.uniform_filter(x,(k,k))
    L=np.array([[1,1,1],[1,-8,1],[1,1,1]])
    y=ndi.correlate(x_smooth,L)
    
    plt.subplot(1,3,2)
    plt.imshow(y,clim=[0,255],cmap='gray')
    
    y_ed=edge_detection(x,5,1.15)
    plt.subplot(1,3,3)
    plt.imshow(y_ed,clim=[0,1],cmap='gray')    
    
if __name__=='__main__':
    main()
    plt.show()


