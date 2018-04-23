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


import matplotlib.pyplot as plt

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


#coords = [ [30,37,32], [40,40,32], [28,41,31], [27,42,31], [58,58,25], [58,48,23] ]
xy, yx = np.meshgrid(np.arange(30,40 ), np.arange(30,40) )
coords = list(  np.concatenate(( xy.reshape([xy.size,1]), yx.reshape([xy.size,1]), 32*np.ones([xy.size,1]) ) ,axis=1) )

#%% single slice algorithm
print('Running algorithm on single voxel')
ck = range(len(coords))
xk = range(len(coords))
atomSpecs = range(len(coords))
for ii in range(len(coords)):
    xx = int(coords[ii][0])
    yy = int(coords[ii][1])
    zz = int(coords[ii][2])
    anatMask = np.ones(dataSize[:3])
    start_time = time.time()
    
    # generate SSFP params
    K, L, E1, E2 = computeAllSSFPParams(t1wimg[xx,yy,zz], t2wimg[xx,yy,zz], TR, alpha, M0, N, numDiff)
    KL = K*L 
    
    KL = KL * np.mean( dataSSFP[xx,yy,zz,ixB0], axis=0 ) / np.sum(KL[0,:])
    
    # create initial impulse
    impulsePrev, atomSpecsInit = generateTensorAtomsFromParam( diffGradients, TRsampling, qvalues, np.zeros([2,1]),np.linspace(1e-4,1e-2,20), np.ones([1]))
    impulsePrev = impulsePrev.dot(KL.T)
    
    # start atom search minimizaion
    tau = 1.0
    ck[ii], xk[ii], impulsePrev, atomSpecs[ii] = runLTIsysID( dataSSFP[xx,yy,zz,:], KL, tau, diffGradients, TRsampling, qvalues, impulsePrev, atomSpecsInit, 1024,20,10)
    
    end_time = time.time()
    print('Total compute time: %s secs' %( end_time - start_time )) 
    
    
    #%% PLOTS
#for ii in range(len(coords)):
#    xx = coords[ii][0]
#    yy = coords[ii][1]
#    zz = coords[ii][2]
    
    minWeight = 0.02
    
    plt.figure()
    plt.plot(ck[ii])
    plt.title('Weights')
    
    temp = np.zeros(dataSize[:2])
    temp[xx,yy] = 1000
    plt.figure()
    plt.subplot(131)
    plt.imshow(dataSSFP[:,:,zz,0].T + temp.T)
    ax = plt.gca()
    for jj in np.where(ck[ii]>minWeight)[0]:
        ang = atomSpecs[ii][jj]['Angles']
        X = np.cos(ang[0])*np.sin(ang[1])
        Y = np.sin(ang[0])*np.sin(ang[1])
        ax.quiver(xx,yy,X,Y,angles='xy',scale=0.5*1/ck[ii][jj])
    plt.title('Fascicle directions axial, vx[%d,%d]' % (xx,yy))
    plt.subplot(132)
    temp = np.zeros([dataSize[0],dataSize[2]])
    temp[xx,zz] = 1000
    plt.imshow(dataSSFP[:,yy,:,0].T + temp.T)
    ax = plt.gca()
    for jj in np.where(ck[ii]>minWeight)[0]:
        ang = atomSpecs[ii][jj]['Angles']
        X = np.cos(ang[0])*np.sin(ang[1])
        Z = np.cos(ang[1])
        ax.quiver(xx,zz,X,Z,angles='xy',scale=0.5*1/ck[ii][jj])
    plt.title('Fascicle directions coronal, vx[%d,%d]' % (xx,zz))
    plt.subplot(133)
    temp = np.zeros([dataSize[1],dataSize[2]])
    temp[yy,zz] = 1000
    plt.imshow(dataSSFP[xx,:,:,0].T + temp.T)
    ax = plt.gca()
    for jj in np.where(ck[ii]>minWeight)[0]:
        ang = atomSpecs[ii][jj]['Angles']
        Y = np.sin(ang[0])*np.sin(ang[1])
        Z = np.cos(ang[1])
        ax.quiver(yy,zz,Y,Z,angles='xy',scale=0.5*1/ck[ii][jj])
    plt.title('Fascicle directions sagital, vx[%d,%d]' % (yy,zz))
    
    np.savez( './resultsVXOpt', coords=coords, xk=xk, ck=ck, aspecs=atomSpecs )
    
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
