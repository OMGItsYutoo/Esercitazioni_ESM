import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.util import random_noise

def smf(x,k,T):
    m=ndi.median_filter(x,(k,k))
    mask=np.abs(m-x)>T
    return (1-mask)*x+mask*m

def mse(x,y):
    return np.mean((x-y)**2)

def psnr(mse):
    return 10*np.log10(255**2/mse)

def main():
    plt.close('all')

    x=np.float64(io.imread("images/lena.jpg"))
    
    noisy=random_noise(x/255,mode='s&p',amount=0.2)*255
    
    ks=[3,5,7,9,11]
    
    best_T = {}
    
    for k in ks:
        psnr_list=[]
        t_values = list(range(5, 50, 5))
        for t in t_values:
            y_smf=smf(noisy,k,t)
            mse_smf=mse(x,y_smf)
            psnr_smf=psnr(mse_smf)
            psnr_list.append(psnr_smf)
        idx_max=np.argmax(psnr_list)
        best_t = t_values[idx_max]
        max_psnr = psnr_list[idx_max]
        
        best_T[k]=(best_t, max_psnr)
        print(f"Kernel {k}x{k} -> Miglior T: {best_t} (PSNR: {max_psnr:.2f} dB)")
        
    psnr_list_smf=[]
    psnr_list_mf=[] 
    
    for i in range(len(ks)):
        y_smf=smf(noisy,ks[i],best_T[ks[i]][0])
        y_mf=ndi.median_filter(noisy,(ks[i],ks[i]))
        mse_smf=mse(x,y_smf)
        mse_mf=mse(x,y_mf)
        psnr_smf=psnr(mse_smf)
        psnr_mf=psnr(mse_mf)
        psnr_list_smf.append(psnr_smf)
        psnr_list_mf.append(psnr_mf)
        
    plt.figure()
    plt.plot(ks,psnr_list_smf, label='smf filter')
    plt.plot(ks,psnr_list_mf, label='mf filter')
    plt.ylabel('PSNR')
    plt.xlabel('k value')
    plt.grid('on')
    plt.legend()
    
    k=5
    y_smf=smf(noisy,k,best_T[5][0])
    y_mf=ndi.median_filter(noisy,(k,k))
    
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(x,cmap='gray',clim=[0,255])
    plt.subplot(1,4,2)
    plt.imshow(noisy,cmap='gray',clim=[0,255])
    plt.subplot(1,4,3)
    plt.imshow(y_smf,cmap='gray',clim=[0,255])
    plt.subplot(1,4,4)
    plt.imshow(y_mf,cmap='gray',clim=[0,255])
    
if __name__=='__main__':
    main()
    plt.show()