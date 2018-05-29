#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:12:03 2018

@author: ch199899
"""
import numpy as np

from joblib import Parallel, delayed


def equallySpacedSphereSampling(numPhi, numTheta):

    # determine angles
    angles = np.zeros([2, numPhi*numTheta ], dtype=np.float32)
    for ii in range(numPhi):
        phi = 2*np.pi * ( (ii+0.5)/numPhi )
        
        for jj in range(numTheta):
#            theta = 2*np.arcsin( 2*( (jj+0.5)/numTheta -0.5 ) )    # full length from -pi to pi
            theta = 2*np.arcsin( ( (jj+0.5)/numTheta ) )     # half length, from 0 to pi
            
            angles[:,ii*numTheta + jj] = np.array([phi,theta])
    
    # generate vectors    
    vecs = generateVectorFromAngle( angles[0,:], angles[1,:] )
    
    
    return angles, vecs
    
def uniformRAndomDistributionOnSphere(numAng):
    
    phi =  np.random.uniform( 0.0, 2*np.pi,[1,numAng])
    
    th = np.abs( np.arccos(1 - 2*np.random.uniform( 0.0, 1.0,[1,numAng])) - np.pi/2.0 )*2
    
    angles = np.concatenate((phi, th), axis=0)
    
    return angles
    
def generateVectorFromAngle(yaw, pitch):
    
    if isinstance(yaw, np.ndarray):
        M = yaw.size
        
        vec = range(M)
    
        roll = 0
        for mm in range(M):
            Rx = np.matrix([[1, 0, 0],[0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            Ry = np.matrix([ [np.cos(pitch[mm]), 0, np.sin(pitch[mm])], [0, 1, 0], [-np.sin(pitch[mm]), 0, np.cos(pitch[mm])]  ])
            Rz = np.matrix([[np.cos(yaw[mm]), -np.sin(yaw[mm]), 0],[np.sin(yaw[mm]), np.cos(yaw[mm]),0 ], [0,0,1 ]])
            
            R = np.matmul(Rz, np.matmul( Ry, Rx ) )
        
            vec[mm] = np.matmul( np.array(R), np.eye(3) )
        
    else:    
        roll = 0
        Rx = np.matrix([[1, 0, 0],[0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.matrix([ [np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]  ])
        Rz = np.matrix([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw), np.cos(yaw),0 ], [0,0,1 ]])
        
        R = np.matmul(Rz, np.matmul( Ry, Rx ) )
    
        vec = [np.matmul( np.array(R), np.eye(3) )]
    
    return vec



def generateTensorMatrix(sigma, vecs, sigmaScales, angles):
    
    # prealocate data
    S = sigma.shape[0]
    eS = sigmaScales.shape[0]
    V = len(vecs)
    
    D = []
    atomSpecs = []
    atom2Angle = np.zeros([ V, V*eS*S ], dtype=np.uint8)
    # for all fascicles
    for ss in range(S):
        for es in range(eS):
            for vv in range(V):
                # create tensor
                D.append( np.matmul( vecs[vv], np.matmul( np.diag(sigma[ss,:]*sigmaScales[es]), vecs[vv].transpose() ) ) )
                # save atom specifications
                atomSpecs.append( {'Sigma scale' : sigmaScales[es] , 'Sigma' : sigma[ss,:], 'Direction' : vecs[vv][:,0], 'Angles' : angles[:,vv] } )
                # record on angle reconstruction matrix
                atom2Angle[ vv, ss*eS*V + es*V + vv] = 1
    
#    for es in range(eS):
#        D.append( np.diag(sigmaScales[es]*np.ones([3]))  )
#        atomSpecs.append( {'Sigma scale' : sigmaScales[es] , 'Sigma' : [], 'Direction' : [] } )
        
    return D, atomSpecs, atom2Angle




def simulateRealisticExponentialsFromTensorModel(D, grads):
    
    numAng = grads.shape[0]
    M = len(D)
    
    f = np.ndarray([numAng,M])
    for gg in range(numAng):
        for mm in range(M):
            f[gg,mm] = -np.dot(np.dot(grads[gg,:],D[mm]),grads[gg,:])
            
    return f
    

def generateTensorAtoms( diffGradients, numPhi, numTheta, numSigma ):
    
    # define vecotors
    angles, vecs = equallySpacedSphereSampling(numPhi, numTheta)
    
    # define sigmas
    sigmaScales = 10**np.linspace(-4,-3, 5)
#    sigmas = np.array([ [10,7.5,7.5], [10,5,5], [10,2.5,2.5], [10,1.25,1.25] ])
    sigmas = np.array([[10,2.5,2.5], [10,5,5] ])
#    sigmas = np.array([[10,0.1,0.1]])
    
#    sigmas = np.reshape( sigmas,  [ np.prod(sigmas.shape[0:-1]) ,3]  )
    
    # generate tensors
    Datom, atomSpecs, atom2Angle = generateTensorMatrix( sigmas, vecs,sigmaScales, angles)
    
    # generate poleAtoms
    poleAtoms = simulateRealisticExponentialsFromTensorModel( Datom, diffGradients )
    
    return poleAtoms, angles, sigmas, sigmaScales, Datom, vecs, atomSpecs, atom2Angle


def generateImpulseResponses( poles, bval ):
    
    # define
    numAng, numPoles = poles.shape
    
    # prealocate
    impulseResponses = range(numAng)
    for aa in range(numAng):
        
        impulseResponses[aa] =  np.exp( np.outer(poles[aa,:], bval[aa]) )
        
        if aa == 0:
            concatResponse = impulseResponses[0]
        else:
            concatResponse = np.concatenate( (concatResponse, impulseResponses[aa]), axis=1 )
    
    return concatResponse, impulseResponses
    
     
#%%
#    
#   INPUT:
#       - diffGradients - <numGrad,3>double - each row contains a different NORMALIZED diffusion gradient
#        - TR - <1,numTR>double - diffusion time observed by every sample
#                                For regular diffusion numTR=1 and TR = (\Gamma -\delta/3)\\
#                                In SSFP this, numTR = number of longitudonal periods and TR = (n+1)TR, n=[0..T]
#        - qval - <1,numGrad>double - q values that correspond to each diffusion gradient. 
#                                Note that these are assumed squared in the exponential (i.e. equivalent to |q|**2)
#        - angles - <2,numAng>double - angles of the underlying fiber orientations
#        - sigmaScales - <1,numSigma>double - diffusion coefficient of the fibers
#        - eigvalProp - <1,numProp>double - proportion between biggest eigval and orthogonal eigvals
#        
#    OUTPUT:
#        - impulseResponses - <numAtom, numGrad*T>double - each row contains the data that would be generated by a simulated diffusion fibres. The data is ordered in blocks such that there are the numTR samples with varying diffusion times (at a given 1 value) concatenated for all qvalues
#        
def generateTensorAtomsFromParam( diffGradients, TR, qval, angles, sigmaScales, eigvalProp, numThreads=1 ):
    
    # params
    numAng = angles.shape[1]
    numGrad = diffGradients.shape[0]
    numProp = eigvalProp.size
    numSigma = sigmaScales.size
    numTR = TR.size
    
    # init vectors for diffusion directions
    vecs = generateVectorFromAngle( angles[0,:], angles[1,:] )

    # prealocate diffusion tensors along gradients
    K = Parallel(n_jobs=numThreads)( delayed(np.matmul)(diffGradients,vecs[aa]) for aa in range(numAng) )
    
    # define poles
    sigmas = np.array([ np.ones([numProp]), 1/eigvalProp, 1/eigvalProp ])
    Ks = Parallel(n_jobs=numThreads)( delayed(computeNormalizedTensor)(K[aa], sigmas, numGrad, numProp) for aa in range(numAng) )
    
    diffCoef = np.zeros([numGrad,numAng,numProp,numSigma])
    for aa in range(numAng):
        diffCoef[:,aa,:,:] = np. outer( Ks[aa], sigmaScales ).reshape( [numGrad, numProp, numSigma] )

    atomSpecs = []
    for aa in range(numAng):
        for ii in range(numProp):
            # apply diffusion coefficient
            for ss in range(numSigma):
                # compute diffusion coeff at each gradient directiom
                atomSpecs.append( {'sigmas' : sigmas[:,ii]*sigmaScales[ss], 'angles' : angles[:,aa] } )
                
    diffCoef = np.reshape(diffCoef, [numGrad,numSigma*numProp*numAng])            

    # compute impulse response
#    impulseResponses = np.exp( np.outer( diffCoef ,  np.outer( TR, qval ) ) )
    tempImp = Parallel(n_jobs=numThreads)( delayed(computeDecayingExponentials)( diffCoef[gg,:] ,  TR, qval[gg] ) for gg in range(numGrad) )
    
    impulseResponses = np.array(tempImp).transpose(0,2,1).reshape( [numGrad*numTR,numAng*numProp*numSigma] ).T
    
    return impulseResponses, atomSpecs

def computeDecayingExponentials( diffCoef, TR, qval ):
    return  np.exp( np.outer( diffCoef ,  TR.T*qval ) )
    
def computeNormalizedTensor(K, sigmas, numGrad, numProp):

    Ks = np.ndarray([numGrad,numProp])
    for ii in range(numProp):
        # compute normalized tensor
        Ks[:,ii] = -1.0* np.sum( np.matmul(np.diag(np.sqrt(sigmas[:,ii])), K.T )**2 , axis=0)

    return Ks

def parseAtomSpecs(atomSpecs):
    
    epsilonSig = 1e-6
    epsilonAng = 3/180*np.pi
    numAtoms = len(atomSpecs)
    
    angles = atomSpecs[0]['angles'].reshape([1,2])
    sigmas = atomSpecs[0]['sigmas'].reshape([1,3])
    allFA = np.zeros([numAtoms], dtype = np.float32)
    allFA[0] = sigmas[0,0]/sigmas[0,1]
    allSig = np.zeros([numAtoms], dtype = np.float32)
    allSig[0]= sigmas[0,0]
    allAng = np.zeros([numAtoms,2], dtype = np.float32)
    allAng[0,:] = angles
    
    for aa in range(1,numAtoms):
        if np.all( np.sum((angles - np.tile( atomSpecs[aa]['angles'] ,[angles.shape[0],1]))**2, axis=1) > epsilonAng**2 ):
            angles = np.concatenate( ( angles, atomSpecs[aa]['angles'].reshape([1,2]) ) , axis=0) 
        
        if np.all( np.sum((sigmas - np.tile( atomSpecs[aa]['sigmas'] ,[sigmas.shape[0],1]))**2, axis=1) > epsilonSig ):
            sigmas = np.concatenate( ( sigmas, atomSpecs[aa]['sigmas'].reshape([1,3]) ) , axis=0 ) 
            
        allFA[aa] = atomSpecs[aa]['sigmas'][0]/atomSpecs[aa]['sigmas'][1]
        allSig[aa] = atomSpecs[aa]['sigmas'][0]
        allAng[aa,:] = atomSpecs[aa]['angles']
            
    return angles, sigmas, allFA, allSig, allAng
    
#%%
def generateTensorAtomsFromAtomSpecs( diffGradients, bval, atomSpecs , TR=np.ones([1]) ,numThreads=1):

    # parse atomSpecs
    numGrad = bval.size
    numAtom = len(atomSpecs)
    numTR = TR.size
    sink1, sink2, eigvalProp, sigmaScales, angles = parseAtomSpecs(atomSpecs)

    # init vectors for diffusion directions
    vecs = generateVectorFromAngle( angles[:,0], angles[:,1] )

    # prealocate diffusion tensors along gradients
    K = Parallel(n_jobs=numThreads)( delayed(np.matmul)(diffGradients,vecs[aa]) for aa in range(numAtom) )
    
    # define poles
    sigmas = np.array([ np.ones([numAtom]), 1/eigvalProp, 1/eigvalProp ])
    Ks = Parallel(n_jobs=numThreads)( delayed(computeNormalizedTensor)(K[aa], sigmas[:,aa].reshape(3,1), numGrad, 1) for aa in range(numAtom) )
    
    # diffCoef = np.zeros([numGrad, numAtom])
    # for aa in range(numAtom):
    #     diffCoef[:,aa] = (Ks[aa]* sigmaScales[aa]).reshape(numGrad)
        
    diffCoef = np.zeros([numGrad,1,1,numAtom])
    for aa in range(numAtom):
        diffCoef[:,:,:,aa] = np. outer( Ks[aa], sigmaScales[aa] ).reshape( [numGrad, 1, 1] )
    diffCoef = np.reshape(diffCoef, [numGrad,numAtom])

    # compute impulse response
#    impulseResponses = np.exp( np.outer( diffCoef ,  np.outer( TR, qval ) ) )
    tempImp = Parallel(n_jobs=numThreads)( delayed(computeDecayingExponentials)( diffCoef[gg,:] ,  TR, bval[gg] ) for gg in range(numGrad) )
    
    impulseResponses = np.array(tempImp).transpose(0,2,1).reshape( [numGrad*numTR,numAtom] ).T
    
    return impulseResponses


def simulateMultiShellHardi( numAngles, bvalues ):

    sink, vecs = equallySpacedSphereSampling(numAngles[0], numAngles[1])

    diffGradients = np.zeros([0,3])
    allBvalues = []
    maxBvalue = np.max(bvalues)
    for bb in bvalues:
        for vv in vecs:
            diffGradients = np.concatenate(( diffGradients, vv[:,0].reshape(1,3)/float(maxBvalue)*bb ),axis=0)
            allBvalues.append(bb)

    return diffGradients, np.array(allBvalues)

