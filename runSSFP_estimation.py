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
from LTI_systemIdentification import runLTIsysIDonSlice,  clusterDatapoints, runLTIsysIDonClusters, extrapolateDWIData
from MFMatomGeneration import generateTensorAtomsFromParam, generateTensorAtomsFromAtomSpecs, uniformRAndomDistributionOnSphere, simulateMultiShellHardi, generateVectorFromAngle


# import matplotlib.pyplot as plt

#%% set paths
baseFolder = './data_DWI/CSRBRAINS/20180416/'
ssfpFolder = baseFolder + 'data_for_analysis/ssfp/'
header = 'ssfpData.nhdr'
t1wPath = baseFolder + 'data_for_analysis/bestt1w_lowres.nrrd'
t2wPath = baseFolder + 'data_for_analysis/bestt2w_lowres.nrrd'
maskPath = baseFolder + 'common-processed/manualMaskssfp.nrrd'

appendName = 'basicAtom_ssfp'
saveFolder = 'data_DWI/CSRBRAINS/20180416/common-processed/ssfp/'

#%% set params
alpha = 35  # degrees (flip angle)
M0 = 1  # T   (magnetic field strength)
G = 40  # mT/m   (max gradient strength)
N = 5   # number of longitudonal lines (echo number +1)
TR = 40*1e-3 #s

numThreads = 50

#%% load data
dataSSFP, qvalues, diffGradients, Bmax, headerPath, uniqueGrads, valIX = loadDWIdata(ssfpFolder, header)

t1wimg = nrrd.read(t1wPath)[0]
t2wimg = nrrd.read(t2wPath)[0]

qvalues = np.abs(qvalues)**2

#%% Compute params
dataSize = dataSSFP.shape
numDiff = diffGradients.shape[0]
numBval = qvalues.size 
spacing = Bmax/float(numBval-1)
ixB0 = np.where( qvalues == 0 )[0]  # Find B-balues equal to 0

TRsampling = (np.arange(N) +1)*TR


# Retrieve anatomical mask
try:
    anatMask = nrrd.read( maskPath )[0]
except:
    anatMask = np.ones(dataSize[:-1])
    print 'Anatomical mask not found'


#%% single slice algorithm
print('Running algorithm on clusters voxel')
start_time = time.time()

# compute SSFP variables
K, L, E1, E2 = computeAllSSFPParams(t1wimg, t2wimg, TR, alpha, M0, N)
KL = np.tile(K,[N,1,1,1]).transpose(1,2,3,0)*L
KL = KL * np.tile( np.mean( dataSSFP[:,:,:,ixB0], axis=3 )/ np.sum(KL,axis=3) ,[    N,1,1,1]).transpose(1,2,3,0) 
KL = KL.reshape(np.prod(dataSize[:-1]),N)

# eliminate the ones that give unreasonable values
badVx =  np.where( np.isnan( KL ))[0]
anatMask = anatMask.ravel()
anatMask[badVx] = 0

# eliminate the background points
noiseVx = np.where(  np.max(dataSSFP[:,:,:, qvalues > 0],axis=3)  > np.mean( dataSSFP[:,:,:,ixB0] , axis=3) )[0]
anatMask[noiseVx] = 0
anatMaskIX = np.where(anatMask.ravel())[0]

# create clusters of data
print 'Generating clusters'
clusterGroups = clusterDatapoints( KL, anatMask, 10000 )

# run algorithm
print('start!')
recData, atomCoef, atomSpecs =  runLTIsysIDonClusters( dataSSFP, KL, anatMask, clusterGroups, diffGradients, TRsampling, qvalues,  ixB0 , 1024,10,10, numThreads)
end_time = time.time()
print('Total compute time: %s secs' %( end_time - start_time )) 
    
# save output
np.savez( 'data_DWI/resultsVXOpt',  atomCoef=atomCoef, aspecs=atomSpecs, clusterIX=clusterGroups  )


# generate SSFP reconstructions
tempRec = np.zeros([np.prod(dataSize[:-1]), dataSize[-1]])
for ii in range(len(clusterGroups)):
    tempRec[ anatMaskIX[clusterGroups[ii]] ,:] = recData[ii].T

tempRec = tempRec.reshape(dataSize) 
saveNRRDwithHeader( np.float32(tempRec), headerPath, saveFolder, 'recDataSSFP' , qvalues, diffGradients )

# generate DWI data
dwiGrad, dwiBvals = simulateMultiShellHardi([8,4],[0,250,500,1000,1500,2000])
dwiData = extrapolateDWIData( atomCoef, atomSpecs, anatMaskIX, dwiGrad, dwiBvals, dataSize, clusterGroups)
dwiData = dwiData * np.tile(np.mean( dataSSFP[:,:,:,ixB0], axis=3 ),[dwiBvals.size,1,1,1]).transpose(1,2,3,0)

saveNRRDwithHeader( np.float32(dwiData), headerPath, saveFolder, 'dwiRecSSFP' , dwiBvals, dwiGrad )