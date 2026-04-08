import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

def mean_squared_error(x, y):
    return np.mean((x - y)**2)

def adaptive_filter(noisy_img, noise_var, window_size=7):
    mu_l = ndi.uniform_filter(noisy_img, size=window_size)
    sq_mu_l = ndi.uniform_filter(noisy_img**2, size=window_size)
    var_l = sq_mu_l - mu_l**2
    
    var_l = np.maximum(var_l, noise_var)
    
    denoised = noisy_img - (noise_var / var_l) * (noisy_img - mu_l)
    return denoised

def main():

    x = np.float64(io.imread("images/barbara.png"))
    
    sigmas = np.arange(5, 20, 5)
    mse_arithmetic = []
    mse_adaptive = []

    for s in sigmas:
        noise_var = s**2
        noise = s * np.random.randn(x.shape[0],x.shape[1])
        noisy_img = x + noise
        
        img_arithmetic = ndi.uniform_filter(noisy_img, size=7)
        mse_arithmetic.append(mean_squared_error(x, img_arithmetic))
        
        img_adaptive = adaptive_filter(noisy_img, noise_var, window_size=7)
        mse_adaptive.append(mean_squared_error(x, img_adaptive))

    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, mse_arithmetic, 'r-o', label='Media Aritmetica (7x7)')
    plt.plot(sigmas, mse_adaptive, 'b-s', label='Filtro Adattivo')
    
    plt.title('Confronto Performance: Media vs Adattivo')
    plt.xlabel('Deviazione Standard del rumore ($\sigma$)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(noisy_img, clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(img_adaptive, clim=[0,255],cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(img_arithmetic, clim=[0,255],cmap='gray')
    

if __name__ == '__main__':
    main()
    plt.show()
    