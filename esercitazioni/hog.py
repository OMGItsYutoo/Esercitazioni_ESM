import skimage.data as data
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.feature import hog
def main():
    x=np.float64(io.imread("images/lena.jpg"))
    
    y, hog_image = hog(x, pixels_per_cell=(16, 16), visualize=True)
    
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(hog_image**(1/2), clim=None, cmap="gray")
    plt.show()

if __name__=='__main__':
    main()
    plt.show()