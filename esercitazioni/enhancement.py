import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def main():
    plt.close('all')
    x=np.float64(io.imread('images/luna.jpg')) #non ha senso la coversione in float
    h = np.array([
        [0, 0,  1, 0, 0],
        [0, 0,  0, 0, 0],
        [1, 0, -4, 0, 1],
        [0, 0,  0, 0, 0],
        [0, 0,  1, 0, 0]
    ])
    lap=ndi.correlate(x,h)
    y=x-lap
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,cmap='gray',clim=[0,255])
    plt.subplot(1,3,2)
    plt.imshow(lap,cmap='gray',clim=[0,255])
    plt.subplot(1,3,3)
    plt.imshow(y,cmap='gray',clim=[0,255])
    
if __name__=='__main__':
    main()
    plt.show()