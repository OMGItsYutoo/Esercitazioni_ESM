import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
import skimage.util as util
from skimage.exposure import histogram

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/auto.jpg"))/255
    
    plt.figure()
    plt.imshow(x)
    
    Q=[1,10,90]
    k=16
    
    x_to_save = util.img_as_ubyte(x)
    
    means=[]
    
    elab_imgs=[]
    
    for q in Q:    
        io.imsave(f"images/auto_q_{q}.jpg",x_to_save,quality=q)
        x_q=np.float64(io.imread(f"images/auto_q_{q}.jpg"))/255
        dq=(x-x_q)**2
        
        y_p=np.mean(dq, axis=2)        
        
        y=ndi.uniform_filter(y_p,(k,k))
        
        means.append(np.mean(y))
        elab_imgs.append(y)

    x_min=elab_imgs[np.argmin(means)]
   
    T=1.5e-5
    mask=x_min<T
    
    plt.figure()
    plt.imshow(mask,clim=[0,1],cmap='gray')
    plt.title('Mappa di Contraffazione')
    
if __name__=='__main__':
    main()
    plt.show()