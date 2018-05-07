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
from LTI_systemIdentification import runLTIsysIDonSlice,  clusterDatapoints, runLTIsysIDonClusters
from MFMatomGeneration import generateTensorAtomsFromParam, generateTensorAtomsFromAtomSpecs, uniformRAndomDistributionOnSphere, generateVectorFromAngle


# import matplotlib.pyplot as plt

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
M0 = 1  # T   (magnetic field strength)
G = 40  # mT/m   (max gradient strength)
N = 5   # number of longitudonal lines (echo number +1)
TR = 40*1e-3 #s

numThreads = 1

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
    anatMask = np.ones(dataSize)
    print 'Anatomical mask not found'


#%% single slice algorithm
print('Running algorithm on clusters voxel')
start_time = time.time()

# compute SSFP variables
K, L, E1, E2 = computeAllSSFPParams(t1wimg, t2wimg, TR, alpha, M0, N)
KL = np.tile(K,[N,1,1,1]).transpose(1,2,3,0)*L
KL = KL * np.tile( np.mean( dataSSFP[:,:,:,ixB0], axis=3 )/ np.sum(KL,axis=3) ,[    N,1,1,1]).transpose(1,2,3,0) 

# create clusters of data
clusterGroups = clusterDatapoints( t1wimg, t2wimg, 5000 )

# run algorithm
print('start!')
recData, atomCoef, atomSpecs =  runLTIsysIDonClusters( dataSSFP, KL, anatMask, clusterGroups, diffGradients, TRsampling, qvalues,  ixB0 , 1024,10,10, numThreads)
end_time = time.time()
print('Total compute time: %s secs' %( end_time - start_time )) 
    
np.savez( 'data_DWI/resultsVXOpt',  xk=recData, atomCoef=atomCoef, aspecs=atomSpecs )

# #%% PLOTS
# medXX = int(np.median(xx))
# medYY = int(np.median(yy))
# medZZ =  int(np.median(zz))

# xx = xx.ravel()
# yy = yy.ravel()
# zz = zz.ravel()

# scaleFactor = 1

# minWeight = 0.02

# plt.figure()
# plt.plot(atomCoef)
# plt.title('Weights')

# plt.figure()
# temp = np.zeros(dataSize[:2])
# temp[xx,yy] = 1000
# plt.subplot(131)
# plt.imshow(dataSSFP[:,:,medZZ,0].T + temp.T)
# ax = plt.gca()
# for ii in range(1000):
#     for jj in np.where(atomCoef[:,ii]>minWeight)[0][1:]:
#         ang = atomSpecs[jj]['angles']
#         X = np.cos(ang[0])*np.sin(ang[1])
#         Y = np.sin(ang[0])*np.sin(ang[1])
#         ax.quiver(coords[ii,0],coords[ii,1],X,Y,scale=scaleFactor/atomCoef[jj,ii], headwidth=1, angles='xy')
# #plt.title('Fascicle directions axial, vx[%d,%d]' % (medXX,medYY))
# plt.subplot(132)
# temp = np.zeros([dataSize[0],dataSize[2]])
# temp[xx,zz] = 1000
# plt.imshow(dataSSFP[:,medYY,:,0].T + temp.T)
# ax = plt.gca()
# for ii in range(1000):
#     for jj in np.where(atomCoef[:,ii]>minWeight)[0][1:]:
#         ang = atomSpecs[jj]['angles']
#         X = np.cos(ang[0])*np.sin(ang[1])
#         Z = np.cos(ang[1])
#         ax.quiver(coords[ii,0],coords[ii,2],X,Z,scale=scaleFactor/atomCoef[jj,ii], headwidth=1, angles='xy')
# #plt.title('Fascicle directions coronal, vx[%d,%d]' % (medXX,medYY))
# plt.subplot(133)
# temp = np.zeros([dataSize[1],dataSize[2]])
# temp[yy,zz] = 1000
# plt.imshow(dataSSFP[medXX,:,:,0].T + temp.T)
# ax = plt.gca()
# for ii in range(1000):
#     for jj in np.where(atomCoef[:,ii]>minWeight)[0][1:]:
#         ang = atomSpecs[jj]['angles']
#         Y = np.sin(ang[0])*np.sin(ang[1])
#         Z = np.cos(ang[1])
#         ax.quiver(coords[ii,1],coords[ii,2],Y,Z,scale=scaleFactor/atomCoef[jj,ii], headwidth=1, angles='xy')
# #plt.title('Fascicle directions sagital, vx[%d,%d]' % (medYY,medZZ))


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
