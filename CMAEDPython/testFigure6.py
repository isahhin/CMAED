# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:52:43 2019

@author: Hasan
"""

#========================================================================================================================
# Python code for CMAED 2019 paper
# Copyright: Sahin ISIK, 2019
#
# link: https://github.com/isahhin/cmaed
# It is restricted to use for personal and scientific research purpose only
# No Warranty
#       (1) "As-Is". Unless otherwise listed in this agreement, this SOFTWARE PRODUCT is provided "as is," with all faults, defects, bugs, and errors.
#       (2 )No Warranty. Unless otherwise listed in this agreement.
# Please cite the following paper when used this code:
#   1. Işık, Şahin, and Kemal Özkan. "Common matrix approach-based multispectral image fusion and its application to edge detection." 
#      Journal of Applied Remote Sensing 13, no. 1 (2019): 016515.
#========================================================================================================================


import canny_edge_detector as ced
import numpy as np
from scipy import misc
import os
import scipy.io as sio
from scipy import linalg as la
import matplotlib.pyplot as plt
import glob as glob
from scipy import ndimage 
from skimage import filters
from sklearn.preprocessing import minmax_scale

# fname = 'fake_and_real_food';
# fname = 'egyptian_statue';
fname = 'real_and_fake_apples'
name = 'complete_ms_data'


if 1: #compute gradients
    print('---------Save Data step-------------')
    
    oriFiles = os.listdir('database/'+name+'/')  # the folder in which our images exists
    filenameOI = 'database/'+ name+ '/'+ fname + '_ms/'+  fname+ '_ms'
   
    gt= misc.imread('database/'+name+'/'+fname+'_ms/'+fname+'_ms/'+fname+'_RGB.bmp')
    plt.imshow(gt)
    plt.show()
    

    imgsFilesPNG = glob.glob(os.path.join(filenameOI,  '*.png'))
    imagePNG = imgsFilesPNG[0]
    imgData = np.float32(misc.imread(imagePNG))
    imgData = imgData[:,9:]
  
    #figure;imshow(imgData,[])
    noOfSamples = len(imgsFilesPNG);
    h, w = imgData.shape
    dataSet = np.zeros( (h, w, noOfSamples), dtype='float32')
    magnitudes = np.zeros( (h, w,noOfSamples), dtype='float32')
    Cmag = np.zeros( (h*w,noOfSamples), dtype='float32')
    
    #compute magnitudes of images
    sigma=2; #image smoothing parameter
    gauss_kernel = [ 0.0002,0.0010, 0.0045, 0.0159, 0.0431, 0.0913, 0.1506, 0.1933, 0.1933, 0.1506, 0.0913, 0.0431, 0.0159, 0.0045, 0.0010, 0.0002]
    deriv_gauss_kernel = [ 0.0043, 0.0113, 0.0384, 0.0997, 0.1949, 0.2775, 0.2635,0.1105,-0.1105,-0.2635,-0.2775,-0.1949,-0.0997,-0.0384,-0.0113,  -0.0043];
    gauss_kernel = np.array(gauss_kernel)
    deriv_gauss_kernel = np.array(deriv_gauss_kernel)
    for ii in range(0,len(imgsFilesPNG) ):
         imagePNG = imgsFilesPNG[ii];
         img = np.float32( misc.imread(imagePNG) );
         img = img[:,9:]/np.max(img.flatten())  #normalizing  
         dataSet[:,:,ii] = img

         gx = ndimage.filters.convolve1d(img, gauss_kernel.T, axis=0, mode='nearest')
         gx = ndimage.filters.convolve1d(gx, deriv_gauss_kernel, axis=1, mode='nearest')
         
         gy = ndimage.filters.convolve1d(img, gauss_kernel, axis=1, mode='nearest')
         gy = ndimage.filters.convolve1d(gy, deriv_gauss_kernel.T, axis=0, mode='nearest')         
         
         mag = np.hypot(gx, gy)

         magnitudes[:,:,ii] = np.float32(mag)
         Cmag[:, ii] = mag.flatten('F')
    
    #sio.savemat('Cmag',  {'Cmag':Cmag}, appendmat=True)      
    meanref = np.mean(Cmag,axis=1) #meanref : mean of magnitude
    referenceMag = Cmag[:,-1] # takes the last one as reference
    
    #refmag : reference magnitude image after mean removal
    refmag = referenceMag - meanref
    B = np.zeros( (h*w, noOfSamples-1), dtype='float32') # B: difference subspace
    for i in range(0, noOfSamples-1):
        B[:,i] = Cmag[:,i] - meanref - refmag
        
   
    # gram schmidt orthogonalization on difference subspace (B)
    u, s = la.qr(B, mode='economic')
    
    
    #difference vector assoicated with reference magnitude (refmag)
    diffMag=0*refmag; 
    
    for ii in range(0, noOfSamples-1):        
        diffMag = diffMag + np.dot(u[:,ii], refmag)*refmag
  
    
    # common magnitude assoicated with reference magnitude (refmag)
    comMag = refmag - diffMag + meanref
    Cmag = comMag.reshape(h, w, order='F')
    Dmag = diffMag.reshape(h, w, order='F')
    refmag = refmag.reshape(h, w, order='F')
    sio.savemat('Cmag',  {'Cmag':Cmag}, appendmat=True)
    sio.savemat('Dmag',  {'Dmag':Dmag},  appendmat=True)
    sio.savemat('refmag',  {'refmag':refmag},  appendmat=True)
    sio.savemat('gx',  {'gx':gx}, appendmat=True)
    sio.savemat('gy',  {'gy':gy},  appendmat=True)
    print('---------Save Data completed-------------')
     
else: #load gradinets
    print('Load Data')
    Cmag = sio.loadmat('Cmag.mat')
    gx = sio.loadmat('gx.mat')
    gy = sio.loadmat('gy.mat')
    Cmag = Cmag.get('Cmag')
    gx = gx.get('gx')
    gy = gy.get('gy')
    Cmag = np.abs(Cmag)
    
    
  
        

