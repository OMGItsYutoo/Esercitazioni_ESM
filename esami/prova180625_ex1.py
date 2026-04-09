import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.util import img_as_ubyte
from skimage.transform import warp, AffineTransform

def rotate(x,theta):
    M,N=x.shape
    c_x, c_y = N/2,M/2
    
    A=(AffineTransform(translation=(-c_x, -c_y)) + 
        AffineTransform(rotation=theta) + 
        AffineTransform(translation=(c_x, c_y)))
    
    return warp(x,A,order=1)

def stima(x,y):    
    theta_list=np.arange(0,101,10)
    mse_list=[]
    
    for theta in theta_list:
        yrot=rotate(y,-theta)
        
        mse=np.mean((x-yrot)**2)
        
        mse_list.append(mse)
        
    plt.figure()
    plt.plot(theta_list,mse_list,'r-*')
    plt.xlabel(r"Theta ($\theta$)")    
    plt.ylabel("MSE")    
    
    min=np.argmin(mse_list)
    return theta_list[min]

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/lena.jpg"))/255
    
    M,N=x.shape
    c_x, c_y = N/2,M/2
    
    A=(AffineTransform(translation=(-c_x, -c_y)) + 
        AffineTransform(rotation=50) + 
        AffineTransform(translation=(c_x, c_y)))
    
    xrot=warp(x,A,order=1)
    
    xrot_save=img_as_ubyte(xrot)
    io.imsave("images/lenarot.jpg",xrot_save)
    
    y=np.float64(io.imread("images/lenarot.jpg"))/255
    
    stim=stima(x,y)
    
    yf=rotate(y,-stim)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,1],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(y,clim=[0,1],cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(yf,clim=[0,1],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()