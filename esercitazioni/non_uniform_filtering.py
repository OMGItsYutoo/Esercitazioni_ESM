import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np

def smooth(x):
    h = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)/16
    return ndi.correlate(x,h,mode='reflect')
    
def main():
    plt.close('all')
    x=np.float64(io.imread('images/space.jpg'))
    
    y=smooth(x)
    
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.show()

if __name__=='__main__':
    main()