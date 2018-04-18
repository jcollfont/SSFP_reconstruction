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
baseFolder = 'data_DWI/CSR_2/'
ssfpFolder = baseFolder + 'common-processed/diffusion/cusp90/01-motioncorrected/'
header = 'something.nhdr'
t1wPath = baseFolder + 'common-processed/diffusion/cusp90/03-dwi2t1w_registration/c035_s02_t1w-to-b0.nrrd'
t2wPath = baseFolder + 'common-processed/diffusion/cusp90/03-dwi2t1w_registration/c035_s02_t2w-to-b0.nrrd'
maskPath = baseFolder + 'something.nrrd'

appendName = 'basicAtom_ssfp'
saveFolder = 'data_DWI/CSR_2/common-processed/ssfp/'

#%% set params
alpha = 30
M0 = 3
N = 5
TR = 2

#%% load data
dataSSFP, bvalues, Gnorm, Bmax, headerPath, diffGradients, valIX = loadDWIdata(ssfpFolder, header)

t1wimg = nrrd.read(t1wPath)[0]
t2wimg = nrrd.read(t2wPath)[0]


# Retrieve anatomical mask
try:
    anatMask = nrrd.read( maskPath )[0]
except:
    print 'Anatomical mask not found'


#%% Compute params
K, L, E1, E2 = computeAllSSFPParams(t1wimg, t2wimg, TR, alpha, M0, N)

dataSize = dataSSFP.shape
numAng = diffGradients.shape[0]
numBval = bvalues.size 
spacing = Bmax/float(numBval-1)
ixB0 = np.where( bvalues == 0 )[0]  # Find B-balues equal to 0


#%% ALGORITHM
print('Running algorithm on all voxels')
sliceBlockSize = 10
sliceBlocks = int(np.ceil(dataSize[2]/float(sliceBlockSize)))
recData = range(sliceBlocks)
C_rec = range(sliceBlocks)
atomSpecs = []
impulsePrev = np.zeros([0])
start_time = time.time()
for zz in range(sliceBlocks):
    print 'Block ' + str(zz) + ' of ' + str(sliceBlocks)
    sliceIX = range( zz*sliceBlockSize, min( (zz+1)*sliceBlockSize, dataSize[2] ) )
    recData[zz], C_rec[zz], atomSpecs, impulsePrev = runLTIsysIDonSlice(dataSSFP[:,:,sliceIX,:], anatMask[:,:,sliceIX], diffGradients, bvalues, ixB0, impulsePrev, atomSpecs, 1024, 10, 10)
    
end_time = time.time()
print('Total compute time: %s secs' %( end_time - start_time )) 


#%% save
np.savez( saveFolder + 'recParams_'+ appendName, crec=C_rec, aspecs=atomSpecs )

# reorganize the resulting data
numAtoms = len(atomSpecs)
tempRec = np.zeros(dataSize,dtype=np.float32)
for zz in range(sliceBlocks):
    sliceIX = range( zz*sliceBlockSize, min( (zz+1)*sliceBlockSize, dataSize[2] ) )
    # rec signal
    tempRec[:,:,sliceIX,:] = recData[zz]
 

# save and compute error
saveNRRDwithHeader( np.float32(tempRec), headerPath, saveFolder, 'recData_' + appendName , bvalues, diffGradients )
errData = np.float32(tempRec - dataSSFP)
saveNRRDwithHeader( errData, headerPath, saveFolder, 'errRecData_' + appendName , bvalues, diffGradients )
