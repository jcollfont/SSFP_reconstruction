#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:13:29 2018

@author: jaume
"""
#%% IMPORTS
from joblib import Parallel, delayed
import numpy as np
from scipy import signal, sparse
from scipy.spatial.distance import cdist

import sklearn as sk

#import matplotlib.pyplot as plt
from MFMatomGeneration import generateTensorAtomsFromParam, generateTensorAtomsFromAtomSpecs, uniformRAndomDistributionOnSphere, generateVectorFromAngle
from SSFP_functions import computeAllSSFPParams


#%% run LTI system identification method
#    This function runs the system identification method described in:
#        B. Yilmaz, K. Bekiroglu, C. Lagoa, and M. Sznaier, “A Randomized Algorithm for Parsimonious Model Identification,” IEEE Trans. Automat. Contr., vol. 63, no. 2, pp. 532–539, 2017.
#    It consists on convex optimization of the system using pre-defined atoms and a Frank-Wolfe algorithm.
#    The original paper proposes to use a random set of atoms. Here we use a fixed set
#
#
#
def runLTIsysID( y, L,  tau, diffGradients, TR, qval, impulseResponsePrev, atomSpecs, numAng, numSigma, numEigvalProp, numThreads=1):
    
    verbose = False
    
    N,M = y.shape

    numBlocks = 50   
    maxIter = 2e2
    minIter = 10
    stepEpsilon = 1e-2
    ckEpsilon = 1e-6
    numAtoms = impulseResponsePrev.shape[0]
    
    activeAtomsFixed = np.arange(len(atomSpecs))
    
    # reporting system
#    if verbose:
#        plt.figure()
 
    # compute initial xk estimate
    ck = sparse.lil_matrix((numAtoms,M), dtype=np.float32)
    
    # prediagonalize Tau
    if isinstance(tau, (list, np.ndarray, tuple)):
        sdTau = sparse.diags(tau)
    else:
        tau = np.array(tau)
    
    # set initial guess
    xk = np.float32( sdTau.dot(np.tile( impulseResponsePrev[0,:], [M,1])).transpose() )
    ck[0,:] = tau

    # determine parallelization blocks
    blockSize  = int(np.ceil(M/np.float(numBlocks)))
    print 'Num blocks: %d of size: %d for total of %d voxels' %(numBlocks, blockSize, M)
    blockIX = range(numBlocks)
    for bb in range(numBlocks):    
        blockIX[bb] = range( bb*blockSize, min( (bb+1)*blockSize, M)  )


    # ITERATE  !
    k = 0
    numAtomsPrev = numAtoms
    while True:
        
        # plots
#        if verbose:
#            plt.clf()
#            plt.plot(xk)
#            plt.plot(y)
#            plt.plot( y - xk )
#            plt.title('Function approximation')
#            plt.legend(['xk','y','grad'])
#            plt.pause(0.1)
        
        # random atom generation
#        angles = np.concatenate( ( np.random.uniform( 0.0, 2*np.pi,[1,numAng]), np.random.uniform( 0.0, np.pi,[1,numAng]) ) , axis=0)
        angles = uniformRAndomDistributionOnSphere(numAng)
        sigmaScales = np.random.uniform(1e-4,1e-2,numSigma)
        eigvalProp = np.random.uniform( 1,20, numEigvalProp)
        
        # create new and join with previous impulse responses
        impulseResponse, atomSpecsNew = generateTensorAtomsFromParam( diffGradients, TR, qval, angles, sigmaScales, eigvalProp, 1 ) 
        impulseResponse = np.concatenate( (impulseResponsePrev, sparse.block_diag(np.tile(L,[N,1])).dot(impulseResponse) ) )
        numAtoms = impulseResponse.shape[0]
            
        # compute gradient
        gradF = ( xk - y )  # ommiting a 2* on the gradient to use it later
        
        # compute projection of gradient on all atoms
#        print 'Search for minimum atom descent'
        tempK = Parallel(n_jobs=numBlocks)( delayed(minSearchPolicy)(impulseResponse, gradF[:, bb]) for bb in blockIX )
        
        # reconstruct minK array        
        minK = np.ndarray([M], dtype=tempK[0].dtype)
        for bb in range(numBlocks):
            minK[blockIX[bb]] =  tempK[bb]
            
        # set descent direction
#        print 'Atom generation'
        a = sdTau.dot(impulseResponse[minK,:]) - xk.transpose()
        

        # compute step length
#        print 'Get alpha'
        normA = np.sum(a**2,axis=1)
        flagsNonInv = np.where(normA < 1e-8)[0]
        if flagsNonInv.size > 0:            # in case a'*a = 0
            normA.flags.writeable = True
            normA[flagsNonInv] = 1.0
        
        alpha = np.maximum( np.minimum( -np.sum( (a * gradF.transpose()) ,axis=1) / normA  ,np.ones([M])) ,np.zeros([M]) )
        
        if flagsNonInv.size > 0:            # in case a'*a = 0
            alpha[ flagsNonInv ] = 0
        

        # update xk
#        print 'Update xk'
        x_prev = xk
        xk = xk + sparse.diags(alpha).dot(a.T).T
       
        # update ck
#        print 'update ck'
        ck = ck.multiply(np.tile(1 - alpha, [numAtomsPrev,1]))
        ck = sparse.vstack( [ck, sparse.lil_matrix((len(atomSpecsNew),M))], format='lil',dtype=np.float32)
        tauAlpha = sdTau.dot(alpha)
        ck[(minK,np.arange(M))] +=  tauAlpha
        
        # prune unused ck
        activeAtoms = np.where(np.sum( ck >=  ckEpsilon ,axis=1) > 0)[0]
        activeAtoms = np.unique( np.concatenate(( activeAtomsFixed, activeAtoms )) )
        
        ck =  ck[activeAtoms,:]

        # save used and prune unused atoms
#        print 'update impulse and atomsects'
        impulseResponsePrev = impulseResponse[ activeAtoms,:]
        atomSpecs = list( atomSpecs[i] for i in np.where( activeAtoms < numAtomsPrev)[0] ) 
        atomSpecs += list( atomSpecsNew[i-numAtomsPrev] for i in np.where( activeAtoms >= numAtomsPrev)[0] ) 
        numAtomsPrev = activeAtoms.size
        
        # evaluate fit
        fitErr = np.sum(np.sum( (xk - y)**2 )) / (N*M)
        
        # evaluate convergence and report iteration
        stepSize = np.sum(np.sum( (xk - x_prev)**2 )) / (N*M)
        updateReport = 'Iter: %d. Obj. fun.: %0.6f. Step: %0.6f. NumAtoms: %d, active: %d' %( k,  fitErr,  stepSize, numAtoms, numAtomsPrev)
        print updateReport
        if (k > minIter) & ((k > maxIter) | (stepSize < stepEpsilon)):
            break

        
        # update counter
        k +=1 
        
    # plots
#    if verbose:
#        plt.clf()
#        plt.plot(xk)
#        plt.plot(y,'o-')
#        plt.plot( y - xk )
#        plt.title('Function approximation')
#        plt.legend(['xk','y','grad'])
#        plt.pause(0.2)
    
    # return
    return ck, xk, impulseResponsePrev, atomSpecs

#%%
def minSearchPolicy(impulseResponse, gradF):
    
    # compute projection of gradient on all atoms
    rp = np.float32(np.dot( impulseResponse , gradF ))

    # select minimum projection
    minK = rp.argmin(axis=0)
    
    return minK

#%%
def voxelWiseOpt(impulseResponse, gradF, tau, xk, L):
    
    impulseResponse = sparse.block_diag(np.tile(L,[gradF.shape[0],1])).dot(impulseResponse)
    
    # compute projection of gradient on all atoms
    rp = np.float32( gradF.dot( impulseResponse ))

    # select minimum projection
    minK = rp.argmin(axis=0)
    
    a = tau *impulseResponse[:,minK] - xk
        
    # compute step length
#        print 'Atom norm compute'
    normA = np.sum(a**2)
    flagsNonInv = normA < 1e-8
    if flagsNonInv:            # in case a'*a = 0
        normA = 1.0
    
#        print 'Get alpha'
    alpha = max( min( - a.T.dot( gradF) / normA ,1),0)
    
    if flagsNonInv:            # in case a'*a = 0
        alpha =  0

    return a, alpha, minK
    
#%%
def runLTIsysIDonSlice( dataSSFP, KL, anatMask, diffGradients, TR, qvalues,  ixB0 , impulsePrev, atomSpecs, numAng, numSigma, numEigvalProp, numThreads=1 ):
    
    dataSize = dataSSFP.shape
    np.random.seed()
    
    # atoms for water fraction
    if impulsePrev.size == 0:
        impulsePrev, atomSpecsInit = generateTensorAtomsFromParam( diffGradients, TR, qvalues, np.zeros([2,1]),np.linspace(1e-4,1e-2,20), np.ones([1]), 1)
    
    anatMaskIX = np.where(anatMask.ravel() == 1)[0]
    recData = np.zeros([dataSize[-1],np.prod(dataSize[0:-1])])
    atomCoef = np.zeros( [0,np.prod(dataSize[0:-1])], dtype=np.float32)
    if anatMaskIX.size > 0:
        
        # vectorize and mask data
        dataSSFP = np.reshape(dataSSFP,[np.prod(dataSize[0:-1]),dataSize[-1]]).transpose()
        dataSSFP = dataSSFP[ :, anatMaskIX]
        KL = np.reshape(KL,[np.prod(dataSize[0:-1]), KL.shape[-1]])
        KL = KL[anatMaskIX,:]

        # run algorithm
        tau = np.ones([np.prod(dataSize[0:-1])])
        ck, xk, impulsePrev, atomSpecs = runLTIsysID( dataSSFP, KL, tau, diffGradients, TR, qvalues, impulsePrev, atomSpecsInit, numAng, numSigma, numEigvalProp,numThreads)
        
        # reconstruct data
        recData[:,anatMaskIX] = xk
        recData = np.reshape( recData ,[dataSize[3], dataSize[0], dataSize[1],dataSize[2]]).transpose(1,2,3,0)
        
        # atom coefficients
        numAtoms = ck.shape[0]
        atomCoef = np.zeros( [numAtoms,np.prod(dataSize[0:-1])], dtype=np.float32)
        atomCoef[:,anatMaskIX] = ck.todense()
#        atomCoef =  np.reshape( atomCoef, [numAtoms, dataSize[0], dataSize[1],dataSize[2]] ).transpose(1,2,3,0)
        atomCoef = sparse.lil_matrix(atomCoef)
    else:
        recData = np.reshape( recData ,[dataSize[3], dataSize[0], dataSize[1],dataSize[2]]).transpose(1,2,3,0)
        
    return recData, atomCoef, atomSpecs, impulsePrev
    

#%%
def computeODFmap( atomCoef, atomSpecs, axisAngles, bandwidth, dataSize, numAtoms, sliceBlockSize):
    
    # define params and prealocate
    sliceBlocks = len(atomCoef)
    angDims = [axisAngles[0].size, axisAngles[1].size]
    numAng = np.prod(angDims)
    
    # define angle vectors
    angles = np.array(np.meshgrid(axisAngles[0],axisAngles[1])).reshape([2, numAng ]).T
    vec = generateVectorFromAngle( angles[:,0], angles[:,1] )
    diffGradients = np.zeros([numAng,3])
    for aa in range(numAng):
        diffGradients[aa,:] = vec[aa][:,0]
    
    # run pairwise distances for each atom
    chi2 = np.ndarray( [numAng,numAtoms] ,dtype=np.float32)
    for aa in range(numAtoms):
        if atomSpecs[aa]['sigmas'][0] != atomSpecs[aa]['Angles'][1]: 
            chi2[:,aa] = cdist(angles, np.array([atomSpecs[aa]['Angles']]))[:,0] ** 2
        

    # generate ODF map
    angleODFmap = np.zeros( [dataSize[0],dataSize[1],dataSize[2], axisAngles[0].size*axisAngles[1].size] ,dtype=np.float32)
    for zz in range(sliceBlocks):
        numAtomsZZ, M = atomCoef[zz].shape
        filledCoef = sparse.vstack( [atomCoef[zz] , sparse.csc_matrix((numAtoms-numAtomsZZ,M))], format='csc',dtype=np.float32)
        tempODFmap = filledCoef.T.dot( np.exp(-.5 * chi2.T / bandwidth**2 ) )
        
        sliceIX = range( zz*sliceBlockSize, min( (zz+1)*sliceBlockSize, dataSize[2] ) )
        angleODFmap[:,:,sliceIX,:] = np.reshape(tempODFmap.T, [ axisAngles[0].size*axisAngles[1].size, dataSize[0],dataSize[1],sliceBlockSize] ).transpose(1,2,3,0)

    # return
    return angleODFmap, diffGradients

#%%
def extrapolateData( atomCoef, atomSpecs, anatMask, diffAngles, bvalues, dataSize, sliceBlockSize):
    
    sliceBlocks = len(atomCoef)

    # impulse responses
    impulseResponse, diffusionGradients = generateTensorAtomsFromAtomSpecs( diffAngles, bvalues, atomSpecs )
    
    # compute extrapolated data
    numDiff = impulseResponse.shape[1]
    extrapData  = np.zeros( [dataSize[0],dataSize[1],dataSize[2], numDiff] ,dtype=np.float32)
    for zz in range(sliceBlocks):
        
        sliceIX = range( zz*sliceBlockSize, min( (zz+1)*sliceBlockSize, dataSize[2] ) )
        extrapData[:,:,sliceIX,:] = np.reshape(atomCoef[zz].dot( impulseResponse ), [dataSize[0],dataSize[1],sliceBlockSize, numDiff] )
    
    return extrapData, diffusionGradients


#%%
def clusterDatapoints( t1wimg, t2wimg, avrgVoxelsPerCluster ):

    dataSize = t1wimg.size
    numClusters = dataSize/avrgVoxelsPerCluster


    # reorder data
    t1wimg = t1wimg.reshape( np.prod(t1wimg.shape[:-1]), t1wimg.shape[-1] )
    t2wimg = t2wimg.reshape( np.prod(t2wimg.shape[:-1]), t2wimg.shape[-1] )
    X = np.concatenate(( t1wimg, t2wimg ), axis=1)

    # look for clusters with K-means
    km = sk.cluster.KMeans(n_clusters=numClusters, verbose=1, n_jobs=20).fit(X)

    # for each group, retrieve label mask annd create indices mask
    groupIX = range(numClusters)
    for gg in range(numClusters):
        groupIX[gg] = np.where( km.labels_ == gg )[0]

    return groupIX

#%%
#
#
#       groupIX -> group 0  is background
#
def runLTIsysIDonClusters( dataSSFP, KL, anatMask, groupIX, diffGradients, TR, qvalues,  ixB0 , numAng, numSigma, numEigvalProp, numThreads=1 ):
    
    dataSize = dataSSFP.shape
    numGroups = len(groupIX)
    np.random.seed()
    
    # atoms for water fraction
    impulsePrev, atomSpecs = generateTensorAtomsFromParam( diffGradients, TR, qvalues, np.zeros([2,1]),np.linspace(1e-4,1e-2,20), np.ones([1]), 1)
    
    # prepare mask and data
    dataSSFP = dataSSFP.reshape( np.prod(dataSize[0:-1]), dataSize[-1] ).T
    anatMaskIX = np.where(anatMask.ravel() == 1)[0]
    dataSSFP = dataSSFP[anatMaskIX,:]

    KL = np.reshape(KL,[np.prod(dataSize[0:-1]), KL.shape[-1]])
    KL = KL[anatMaskIX,:]

    # prealocate data for results
    recData = np.zeros([dataSize[-1],np.prod(dataSize[0:-1])])
    atomCoef = range(numGroups)

    # for every group
    for gr in range(1,numGroups):       # group 0 is backgrounnd

        # retrieve the data from groups
        dataGroup = dataSSFP[groupIX[gr],:]

        if anatMaskIX.size > 0:

            # determina appropriate values for the SSFP model
            meanKL = np.mean( KL )
            impulsePrev = sparse.block_diag( np.tile(KL,[dataSize[-1],1]) ).dot(impulsePrev)

            # run algorithm
            tau = np.ones([np.prod(dataSize[0:-1])])
            ck, xk, impulseNew, atomSpecsNew = runLTIsysID( dataGroup, meanKL, tau, diffGradients, TR, qvalues, impulsePrev, atomSpecs, numAng, numSigma, numEigvalProp,numThreads)
            
            # reconstruct data
            recData[:,groupIX[gr]] = xk

            # atom coefficients
            numAtoms = ck.shape[0]
            atomCoef[gr] = np.zeros( [numAtoms,np.prod(dataSize[0:-1])], dtype=np.float32)
            atomCoef[gr][:,groupIX[gr]] = ck.todense()
            atomCoef[gr] = sparse.lil_matrix(atomCoef[gr])

        # reshape data
        recData = np.reshape( recData ,[dataSize[3], dataSize[0], dataSize[1],dataSize[2]]).transpose(1,2,3,0)
        

    return recData, atomCoef, atomSpecs, impulsePrev