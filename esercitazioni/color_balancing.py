import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/foto.jpg'))/255
    
    plt.figure()
    plt.imshow(x)
    
    x_cmy=1-x
    C=x_cmy[:,:,0]
    M=x_cmy[:,:,1]
    Y=x_cmy[:,:,2]
    
    a=1.5;b=1;c=1
    
    Cn=C**a
    Mn=M**b
    Yn=Y**c
    y_hsv=np.stack((Cn, Mn, Yn),2)
    y=1-y_hsv
    
    Cn=C**1
    Mn=M**0.8
    Yn=Y**0.8
    z_hsv=np.stack((Cn, Mn, Yn),2)
    z=1-z_hsv
    
    #prove di color balancing
    # c_min=np.min(C)    
    # C_new=C-c_min
    # y_cmy=np.stack((C_new,M,Y),2)
    # y=1-y_cmy
        
    # Mn=M+0.1
    # z_cmy=np.stack((C_new,Mn,Y),2)
    # z=1-z_cmy
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(y)
    plt.subplot(1,2,2)
    plt.imshow(z)

if __name__=='__main__':
    main()
    
    plt.show()
    