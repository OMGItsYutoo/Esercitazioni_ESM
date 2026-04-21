import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from numpy import abs as _abs

def zero_crossing(b, thresh=None):
    """
    Restituisce la mappa dei passaggi per zero.

    Parameters
    ----------
    b : matrice 
    thresh : soglia opsionale

    Returns
    -------
    m : mappa dei zero crossing

    """
    if thresh is None:
        thresh = 0.25 * np.mean(_abs(b))
        print('thresh = ', thresh)
    thresh2 = 2.0 * thresh

    m1 = (b[1:-1,1:-1]<0) & (b[1:-1,2:  ]>0) & (_abs(b[1:-1,1:-1]-b[1:-1,2:  ])>thresh) # H[- +]
    m2 = (b[1:-1,0:-2]>0) & (b[1:-1,1:-1]<0) & (_abs(b[1:-1,0:-2]-b[1:-1,1:-1])>thresh) # H[+ -]
    m3 = (b[1:-1,1:-1]<0) & (b[2:  ,1:-1]>0) & (_abs(b[1:-1,1:-1]-b[2:  ,1:-1])>thresh) # V[- +]
    m4 = (b[0:-2,1:-1]>0) & (b[1:-1,1:-1]<0) & (_abs(b[0:-2,1:-1]-b[1:-1,1:-1])>thresh) # V[+ -]
    m5 = (b[1:-1,1:-1]==0) & ((b[1:-1,0:-2]*b[1:-1,2:  ])<0) & (_abs(b[1:-1,0:-2]-b[1:-1,2:  ])>thresh2) # H[- 0 +]
    m6 = (b[1:-1,1:-1]==0) & ((b[0:-2,1:-1]*b[2:  ,1:-1])<0) & (_abs(b[0:-2,1:-1]-b[2:  ,1:-1])>thresh2) # V[- 0 +]
    
    m = m1 | m2 | m3 | m4 | m5 | m6
    m = np.pad(m, ((1,1),(1,1)), mode='constant', constant_values = False)
    return m
   
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
    lap_at=ndi.correlate(mask_at,L)
    
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