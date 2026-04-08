import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from sklearn.cluster import k_means

def T_opt(x):
    d=np.reshape(x,(-1,1)) #trasforma la matrice in un vettore colonna
    centroids,idx,var_sum=k_means(d,2)
    return np.mean(centroids)

def adpt_thresholding(x,L):
    M,N=x.shape
    num_blocks=M//L
    y=np.zeros((M,N),np.bool_)
    for i in range(num_blocks):
        b=x[i*L:i*L+L,:]
        mask=b>T_opt(b)
        y[i*L:i*L+L,:]=mask
    return y
    
def main():
    plt.close('all')

    x=np.fromfile("images/rice.y",dtype=np.uint8)
    x=np.reshape(x,(256,256))
    x=np.float64(x)

    mask_ideal = np.reshape(np.fromfile('images/rice_bw.y', np.uint8), (256,256))
    
    t=T_opt(x)
    mask=x>t

    plt.figure()
    plt.imshow(mask,clim=[0,1], cmap='gray')
    plt.title('Maschera con Segmentazione Globale')
    
    plt.figure()
    plt.imshow(mask_ideal,clim=[0,1], cmap='gray')
    plt.title('maschera ideale')
    
    acc_list=[]
    list_L = [1,2,4,8,16,32,64,128,256]

    for l in list_L:
        y=adpt_thresholding(x,l)
        acc=np.sum(y==mask_ideal)
        acc_list.append(acc)

    plt.figure()
    plt.plot(list_L, acc_list, '-*')
    plt.grid('on')
    plt.ylabel('pixel corretti')
    
    #ottimo per 16
    l=16
    y=adpt_thresholding(x,l)
    plt.figure()
    plt.title('Maschera ottima')
    plt.imshow(y, cmap='gray',clim=[0,1])
    
if __name__=='__main__':
    main()
    plt.show()