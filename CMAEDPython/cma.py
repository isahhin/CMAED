# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:46:16 2019

@author: Hasan
"""
import tensorflow as tf
import numpy as np
from scipy import linalg as la
from qr_factor  import qr_factorization
from keras.layers.core import Activation, Reshape
from tensorflow.python.keras import backend as K
 
def projection_onto_vector(B, Q):

    m, n = Q.shape[0], Q.shape[1] 
    #m:vector size
    #n: number of vector
    Adiff=0*B    
    for ii in range(n):       
         q = Q[ :, :, ii]
     
         dp = np.dot( q, B, axes=1)
         #print('B', B)
         #print('dp', dp)
         res = dp*q
         #print(res)
         Adiff = Adiff +res
         
    return Adiff

def cva_compute(data):
    #data = tf.keras.backend.eval(data)
#    from tensorflow.python.keras import backend as K
#    sess = K.get_session()
#    data = sess.run(data)
#   print(data)
#    
#    sess = tf.Session()
#    with sess.as_default():
#        print( tf.Variable(data).eval() )
#        
    #print(data)
    
    #print(data)
    n = data.shape[1]
    fs = data.shape[2]
   # print('fs', fs)
    
    data = tf.reshape(data, shape=(n, fs))
    #print('data', data)
    DiffFrms = np.zeros((n-1, fs), dtype="float32")
    
    #DiffFrms =  0*data[:, 0:n-1]
    refIndex = 0 # selecting a reference image
    ref = data[refIndex, : ]
    DiffFrms = data[1:n, :] - ref
    
#    idx = 0;
#    for i in range(n):
#        if i == refIndex:
#            continue
#        
#        DiffFrms[:,idx] = data[:,i] - ref 
#        
#        idx = idx + 1;
    #DiffFrms = tf.reshape(DiffFrms, shape=(-1, n-1, fs))
    #print('DiffFrms', DiffFrms)
 #   DiffFrms = tf.reshape(DiffFrms, shape=(n-1, fs))
#    print(DiffFrms)

    # gram schmidt orthogonalization on difference subspace (B)
    #Q, R = la.qr(DiffFrms, mode='economic')
    Q, R  = tf.linalg.qr( tf.transpose(DiffFrms) ) 
    #print(Q)
    #print(ref)
    #Q, R = qr_factorization(DiffFrms)
    
    # Adiff: The difference vector of associated with reference vector(ref)
    # Acom: The common vector of processed class

    #difference vector assoicated with reference magnitude (refmag)  
    Adiff = projection_onto_vector(ref, Q)
    Acom = ref - Adiff
    #ref2 =Acom + Adiff
    #Acom = tf.convert_to_tensor(Acom)
    Acom = tf.reshape(Acom, shape=(1, 1, fs))

    return Acom