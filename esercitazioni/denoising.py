import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

def mean_squared_error(x,y):
    return np.mean((x-y)**2)

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/lena_forbidd.jpg"))
    if x.ndim == 3:
        x = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.114 * x[:,:,2]
    else:
        x = x

    M,N=x.shape
    
    d=10
    n=d*np.random.randn(M,N)    
    noisy=x+n
    
    denoised=ndi.uniform_filter(noisy,(3,3))
    
    mse=mean_squared_error(x,denoised)
    print(f"MSE: {mse}")
    
    X=np.fft.fftshift(np.fft.fft2(x))
    NOISY=np.fft.fftshift(np.fft.fft2(noisy))
    
    # delta = np.zeros((M, N))
    # delta[M//2, N//2] = 1
    # psf = ndi.gaussian_filter(delta, sigma=0.75)
    # H = np.fft.fftshift(np.fft.fft2(psf))
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(np.log(np.abs(X)+1), clim=None,cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(np.log(np.abs(NOISY)+1), clim=None,cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow(np.log(np.abs(H)+1), clim=None,cmap='gray')
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x, clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(noisy, clim=[0,255],cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(denoised, clim=[0,255],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()

