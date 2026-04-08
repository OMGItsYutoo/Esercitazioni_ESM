import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.util import random_noise

def main():
    plt.close('all')

    x=np.float64(io.imread("images/lena_forbidd.jpg"))
    if x.ndim == 3:
        x = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.114 * x[:,:,2]
    else:
        x = x

    x=x/255
    noisy = random_noise(x, mode='s&p')
    
    k=[5,7,9]
    mse_list=[]
    
    for i in k:
        denoised=ndi.median_filter(x,(i,i))
        mse=np.mean((x-denoised)**2)
        mse_list.append(mse)

    plt.figure()
    plt.plot(k,mse_list,'r-o')
    plt.title('Confronto Performance: k=5,7,9')
    plt.xlabel('Dimensione finestra ($\sigma$)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x, cmap='gray', clim=[0,1])
    plt.subplot(1,3,2)
    plt.imshow(noisy, cmap='gray', clim=[0,1])
    plt.subplot(1,3,3)
    plt.imshow(denoised, cmap='gray', clim=[0,1])
    
if __name__=="__main__":
    main()
    plt.show()
