import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt

def block_fun(x):
    values=x[::8]
    return np.median(values)

def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/aerei.png'))
    
    h=np.zeros((3,3),dtype=np.float64)
    h[0,1]=-1
    h[1,1]=2
    h[2,1]=-1

    y=np.abs(ndi.correlate(x,h))
    y[y>50]=0
    
    es=33*ndi.uniform_filter(y,(1,33)) #1 riga e 16+16+1 colonne
    
    e=es-ndi.median_filter(es,(33,1)) #33 righe e 1 colonna
    
    gh=ndi.generic_filter(e,block_fun,(33,1))
    
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(x,cmap='gray',clim=[0,255])
    plt.subplot(1,4,2)
    plt.imshow(gh,cmap='gray',clim=None)
    
    v=h.T

    yv=np.abs(ndi.correlate(x,v))
    yv[yv>50]=0
    
    es=33*ndi.uniform_filter(yv,(33,1)) #1 colonna e 16+16+1 righe
    
    e=es-ndi.median_filter(es,(1,33)) #33 colonne e 1 riga
    
    gv=ndi.generic_filter(e,block_fun,(1,33))
    
    plt.subplot(1,4,3)
    plt.imshow(gv,cmap='gray',clim=None)
    
    g=gh+gv
    
    plt.subplot(1,4,4)
    plt.imshow(g,cmap='gray',clim=None)
    
if __name__=='__main__':
    main()
    plt.show()