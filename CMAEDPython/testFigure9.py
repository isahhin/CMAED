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
import edge_utils as EU
from skimage import filters

name = 'hyperspectral_dataset'


if 0: #compute gradients
    print('---------Save Data step-------------')
    

    data = sio.loadmat('database/'+name+'/'+'PaviaU.mat')
    #gt = sio.loadmat('database/'+name+'/'+'PaviaU_gt.mat')
    
    data = data.get('paviaU')
   
    gt = misc.imread('database/'+name+'/'+'paviaU_gt.png')
    gt = gt[:,:,0]
    plt.imshow(gt)
    plt.show()
    
    
   
  
    #figure;imshow(imgData,[])
    noOfSamples = data.shape[2]
    h, w = data.shape[0], data.shape[1]
    dataSet = np.zeros( (h, w, noOfSamples-1), dtype='float32')
    magnitudes = np.zeros( (h, w,noOfSamples-1), dtype='float32')
    Cmag = np.zeros( (h*w,noOfSamples-1), dtype='float32')
    
    #compute magnitudes of images
    sigma=1; #image smoothing parameter
    gauss_kernel = [  0.0009,   0.0175,   0.1295,   0.3521,    0.3521,    0.1295,    0.0175,    0.0009]
    deriv_gauss_kernel = [  0.0463,    0.1789,    0.4653,    0.3095,   -0.3095,   -0.4653,   -0.1789,   -0.0463 ]
    gauss_kernel = np.array(gauss_kernel)
    deriv_gauss_kernel = np.array(deriv_gauss_kernel)
    for ii in range(0,noOfSamples-1 ):
        
        img = data[:,:,ii];
        img = img-np.min(img.flatten('F'));
        img = img/np.max(img.flatten('F'))  #normalizing  
        dataSet[:,:,ii] = img

        gx = ndimage.filters.convolve1d(img, gauss_kernel.T, axis=0, mode='nearest')
        gx = ndimage.filters.convolve1d(gx, deriv_gauss_kernel, axis=1, mode='nearest')
         
        gy = ndimage.filters.convolve1d(img, gauss_kernel, axis=1, mode='nearest')
        gy = ndimage.filters.convolve1d(gy, deriv_gauss_kernel.T, axis=0, mode='nearest')         
         
        mag = np.hypot(gx, gy)
        mag = np.float32(mag)
        mag = mag/np.max(mag.flatten('F'))
        magnitudes[:,:,ii] = mag
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


