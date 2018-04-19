#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:39:28 2018

@author: ch199899
"""
#%% IMPORTS
import numpy as np
import time
from scipy import sparse
import nrrd
from loadDWIdata import loadDWIdata, saveNRRDwithHeader
from SSFP_functions import computeAllSSFPParams
from LTI_systemIdentification import runLTIsysIDonSlice


#%% set paths
baseFolder = './data_DWI/CSRBRAINS/20180416/'
ssfpFolder = baseFolder + 'data_for_analysis/ssfp/'
header = 'ssfpData.nhdr'
t1wPath = baseFolder + 'data_for_analysis/bestt1w_lowres.nrrd'
t2wPath = baseFolder + 'data_for_analysis/bestt2w_lowres.nrrd'
maskPath = baseFolder + 'something.nrrd'

appendName = 'basicAtom_ssfp'
saveFolder = '/Users/jaume/Desktop/DWI/csr/common-processed/ssfp/'

#%% set params
alpha = 35  # degrees (flip angle)
M0 = 3  # T   (magnetic field strength)
G = 40  # mT/m   (max gradient strength)
N = 2   # number of longitudonal lines (echo number +1)
TR = 40 #ms

#%% load data
dataSSFP, qvalues, diffGradients, Bmax, headerPath, uniqueGrads, valIX = loadDWIdata(ssfpFolder, header)

t1wimg = nrrd.read(t1wPath)[0]
t2wimg = nrrd.read(t2wPath)[0]

qvalues = np.abs(qvalues)**2

# Retrieve anatomical mask
try:
    anatMask = nrrd.read( maskPath )[0]
except:
    print 'Anatomical mask not found'


#%% Compute params
dataSize = dataSSFP.shape
numDiff = diffGradients.shape[0]
numBval = qvalues.size 
spacing = Bmax/float(numBval-1)
ixB0 = np.where( qvalues == 0 )[0]  # Find B-balues equal to 0

K, L, E1, E2 = computeAllSSFPParams(t1wimg, t2wimg, TR, alpha, M0, N, numDiff)

KL = np.reshape(np.tile(K,[N*numDiff**2,1,1,1]),[numDiff,N*numDiff,dataSize[0],dataSize[1],dataSize[2]]).transpose(2,3,4,0,1)*L

TRsampling = (np.arange(N) +1)*TR



#%% single slice algorithm
print('Running algorithm on single slice')
zz = 31
anatMask = np.ones(dataSize[:3])
start_time = time.time()
recData, C_rec, atomSpecs, impulsePrev = runLTIsysIDonSlice(dataSSFP[:,:,zz,:], KL, anatMask[:,:,zz], diffGradients, TRsampling, qvalues, ixB0, np.zeros([0]), [], 1024, 10, 10)
end_time = time.time()

#%% ALGORITHM
#print('Running algorithm on all voxels')
#sliceBlockSize = 10
#sliceBlocks = int(np.ceil(dataSize[2]/float(sliceBlockSize)))
#recData = range(sliceBlocks)
#C_rec = range(sliceBlocks)
#atomSpecs = []
#impulsePrev = np.zeros([0])
#start_time = time.time()
#for zz in range(sliceBlocks):
#    print 'Block ' + str(zz) + ' of ' + str(sliceBlocks)
#    sliceIX = range( zz*sliceBlockSize, min( (zz+1)*sliceBlockSize, dataSize[2] ) )
#    recData[zz], C_rec[zz], atomSpecs, impulsePrev = runLTIsysIDonSlice(dataSSFP[:,:,sliceIX,:], K*L, anatMask[:,:,sliceIX], diffGradients, bvalues, ixB0, impulsePrev, atomSpecs, 1024, 10, 10)
#    
#end_time = time.time()
#print('Total compute time: %s secs' %( end_time - start_time )) 


#%% save
#np.savez( saveFolder + 'recParams_'+ appendName, crec=C_rec, aspecs=atomSpecs )
#
## reorganize the resulting data
#numAtoms = len(atomSpecs)
#tempRec = np.zeros(dataSize,dtype=np.float32)
#for zz in range(sliceBlocks):
#    sliceIX = range( zz*sliceBlockSize, min( (zz+1)*sliceBlockSize, dataSize[2] ) )
#    # rec signal
#    tempRec[:,:,sliceIX,:] = recData[zz]
# 
#
## save and compute error
#saveNRRDwithHeader( np.float32(tempRec), headerPath, saveFolder, 'recData_' + appendName , bvalues, diffGradients )
#errData = np.float32(tempRec - dataSSFP)
#saveNRRDwithHeader( errData, headerPath, saveFolder, 'errRecData_' + appendName , bvalues, diffGradients )
