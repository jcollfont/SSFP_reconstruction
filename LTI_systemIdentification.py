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

#import matplotlib.pyplot as plt
from MFMatomGeneration import generateTensorAtomsFromParam, generateTensorAtomsFromAtomSpecs, uniformRAndomDistributionOnSphere, generateVectorFromAngle


#%% run LTI system identification method
#    This function runs the system identification method described in:
#        B. Yilmaz, K. Bekiroglu, C. Lagoa, and M. Sznaier, “A Randomized Algorithm for Parsimonious Model Identification,” IEEE Trans. Automat. Contr., vol. 63, no. 2, pp. 532–539, 2017.
#    It consists on convex optimization of the system using pre-defined atoms and a Frank-Wolfe algorithm.
#    The original paper proposes to use a random set of atoms. Here we use a fixed set
#
#
#
def runLTIsysID( y, L,  tau, diffGradients, bval, impulseResponsePrev, atomSpecs, numAng, numSigma, numEigvalProp):
    
    verbose = False
    
    N, M = y.shape
    maxIter = 1e2
    stepEpsilon = 1e-1
    ckEpsilon = 2.5
    numAtoms = impulseResponsePrev.shape[0]
    
    activeAtomsFixed = np.arange(len(atomSpecs))
    
    # reporting system
#    if verbose:
#        plt.figure()
 
    # compute initial xk estimate
    ck = sparse.lil_matrix((numAtoms,M), dtype=np.float32)
    
    # prediagonalize Tau
    sdTau = sparse.diags(tau)
    
    # set initial impulse responses if not available
    if numAtoms == 0:
        impulseResponsePrev = np.float32( L.dot( generateTensorAtomsFromParam( diffGradients, bval, np.zeros([2,1]), np.ones([1])*1e-3, np.ones([1]) ) ) )
    
    # set initial guess
    xk = np.float32( sdTau.dot(np.tile( impulseResponsePrev[0,:], [M,1])).transpose() )
    ck[0,:] = tau
    
    # determine parallelization blocks
    numBlocks = 50
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
        sigmaScales = 10**np.random.uniform(-4,-2,numSigma)
        eigvalProp = np.random.uniform( 1,20, numEigvalProp)
        
        # create new and join with previous impulse responses
        impulseResponse, atomSpecsNew = generateTensorAtomsFromParam( diffGradients, bval, angles, sigmaScales, eigvalProp )
        impulseResponse = np.concatenate( (impulseResponsePrev, np.float32(L.dot(impulseResponse))) )
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
#        print 'Atom norm compute'
        normA = np.sum(a**2,axis=1)
        flagsNonInv = np.where(normA < 1e-8)[0]
        if flagsNonInv.size > 0:            # in case a'*a = 0
            normA.flags.writeable = True
            normA[flagsNonInv] = 1.0
        
#        print 'Get alpha'
        alpha = np.maximum( np.minimum( -np.sum( (a * gradF.transpose()) ,axis=1) / normA  ,np.ones([M])) ,np.zeros([M]) )
        
        if flagsNonInv.size > 0:            # in case a'*a = 0
            alpha[ flagsNonInv ] = 0
        
        # update xk
#        print 'Update xk'
        x_prev = xk
        xk = xk + sparse.diags(alpha).dot(a).transpose()
        
        # update ck
#        print 'update ck'
        ck = ck.multiply(np.tile(1 - alpha, [numAtomsPrev,1]))
        ck = sparse.vstack( [ck, sparse.lil_matrix((len(atomSpecsNew),M))], format='lil',dtype=np.float32)
        tauAlpha = sdTau.dot(alpha)
        ck[(minK,np.arange(M))] +=  tauAlpha
        
        # prune unused ck
#        activeAtoms = np.unique( np.concatenate(( activeAtomsFixed, ck.nonzero()[0] )) )
#        activeAtoms = np.unique( ck.nonzero()[0] ) 
        
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
        fitErr = np.linalg.norm( xk - y  ,ord='fro')**2 / (N*M)
        
        
        # evaluate convergence and report iteration
        stepSize = np.linalg.norm(xk - x_prev, ord='fro')**2 / (N*M)
        updateReport = 'Iter: %d. Obj. fun.: %0.6f. Step: %0.6f. NumAtoms: %d, active: %d' %( k,  fitErr,  stepSize, numAtoms, numAtomsPrev) 
        print updateReport
        if (k > maxIter) | (stepSize < stepEpsilon):
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
def runLTIsysIDonSlice( dataSlice, anatMask, diffGradients, bval,  ixB0 , impulsePrev, atomSpecs, numAng, numSigma, numEigvalProp ):
    
    dataSize = dataSlice.shape
    np.random.seed()
    
    # atoms for water fraction
    if impulsePrev.size == 0:
        impulsePrev, atomSpecs = generateTensorAtomsFromParam( diffGradients, bval, np.zeros([2,1]),10**np.random.uniform(-4,-2,numSigma), np.ones([1]))
    
    anatMaskIX = np.where(anatMask.ravel() == 1)[0]
    recData = np.zeros([dataSize[-1],np.prod(dataSize[0:-1])])
    atomCoef = np.zeros( [0,np.prod(dataSize[0:-1])], dtype=np.float32)
    if anatMaskIX.size > 0:
        
        # vectorize data
        dataSlice = np.reshape(dataSlice,[np.prod(dataSize[0:-1]),dataSize[-1]]).transpose()
        dataSlice = dataSlice[ :, anatMaskIX]
        
        # run algorithm
        tau = np.mean( dataSlice[ixB0,:], axis=0 )
        ck, xk, impulsePrev, atomSpecs = runLTIsysID( np.float32(dataSlice), tau, diffGradients, bval, impulsePrev, atomSpecs, numAng, numSigma, numEigvalProp)
        
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