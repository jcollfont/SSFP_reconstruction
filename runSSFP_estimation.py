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
from LTI_systemIdentification import runLTIsysIDonSlice, runLTIsysID
from MFMatomGeneration import generateTensorAtomsFromParam, generateTensorAtomsFromAtomSpecs, uniformRAndomDistributionOnSphere, generateVectorFromAngle


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
TR = 40*1e-3 #s

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

TRsampling = (np.arange(N) +1)*TR



#%% single slice algorithm
print('Running algorithm on single voxel')
xx = 28
yy = 42
zz = 31
anatMask = np.ones(dataSize[:3])
start_time = time.time()

# generate SSFP params
K, L, E1, E2 = computeAllSSFPParams(t1wimg[xx,yy,zz], t2wimg[xx,yy,zz], TR, alpha, M0, N, numDiff)
KL = L

# create initial impulse
impulsePrev, atomSpecs = generateTensorAtomsFromParam( diffGradients, TRsampling, qvalues, np.zeros([2,1]),10**np.linspace(-5,-2,20), np.ones([1]))
impulsePrev = impulsePrev.dot(KL.T)

# start atom search minimizaion
tau = np.mean( dataSSFP[xx,yy,zz,ixB0], axis=0 )*100
ck, xk, impulsePrev, atomSpecs = runLTIsysID( dataSSFP[xx,yy,zz,:], KL, tau, diffGradients, TRsampling, qvalues, impulsePrev, atomSpecs, 1024,10,10)

end_time = time.time()
print('Total compute time: %s secs' %( end_time - start_time )) 


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
