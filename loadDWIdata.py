#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:48:23 2018

@author: ch199899
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:17:08 2018

@author: ch199899
"""
#import matplotlib.pyplot as plt
import numpy as np
import nrrd
import os
import re




def loadDWIdata(loadFolder, header):
    

    epsilon = 0.99
    
    # load header
    headerPath = loadFolder+header
    hddr = open(headerPath,'r')
    lines = hddr.readlines()
    hddr.close()
    
    # extract gradients and stop when names of files start
    for ii in range(0,len(lines),1):
        if re.match("DWMRI_b-value:=",lines[ii]):
            scaleB = float(lines[ii].split(':=')[1])
            break
    gradient = []
    for jj in range(ii+1,len(lines),1):
        if re.match("DWMRI_gradient",lines[jj]):
            try:
                gradSTR = lines[jj].split(':=')[1].split(' ')[1:4]
                gradient.append( np.array([float(gradSTR[0]), float(gradSTR[1]), float(gradSTR[2])] ))
            except:
                gradSTR = lines[jj].split(':=')[1].split(' ')[0:3]
                gradient.append( np.array([float(gradSTR[0]), float(gradSTR[1]), float(gradSTR[2])] ))
                    
            
        elif lines[jj].find('data file: LIST') > -1:
            break
    
    numGradients = len(gradient)
    
    # rretrieve the diffusion files
    diffFiles = []
    for fi in lines[jj+1:]:
        diffFiles.append(fi[:-1])
            
        
        
    # compute b values
    bvalues = np.ndarray(numGradients)
    Gnorm = np.zeros([numGradients,3])
    for ii in range(0,numGradients,1):
        bvalues[ii] = scaleB*pow(np.linalg.norm(gradient[ii],2),2)
        if bvalues[ii]  != 0:
            Gnorm[ii,:] =  gradient[ii] / np.linalg.norm(gradient[ii],2)
   
    # find redundant gradients and clear them out
    Corre = np.matmul( Gnorm, Gnorm.transpose() )
    ixGen = np.ones([numGradients])
    valIX = -1*np.ones([numGradients])
    for ii in range(numGradients):
        if ixGen[ii] == 1:
            ix = np.where( Corre[ii,:] > epsilon)
            valIX[ix] = ii
            ixGen[ix] = 0
            ixGen[ii] = 1
            
    uniqueGrads = Gnorm[ (ixGen == 1)&(bvalues > 0) ]
    
    # load images
    numImg = numGradients
    nrrdImg = []
    for ii in range(numGradients):
        if diffFiles[ii].find('.nrrd') > 0:
            nrrdImg.append( nrrd.read(loadFolder + diffFiles[ii]) )
            
    
    ## ORGANIZE DATA        
    numImg = len(nrrdImg)
    numX, numY, numZ = nrrdImg[0][0].shape
    bvalNrrd = np.ndarray([numX, numY,numZ,numImg], dtype=np.float32)
    
    for img in range(numImg):
        bvalNrrd[:,:,:,img] = nrrdImg[img][0]
    
#    plt.plot(np.reshape(bvalNrrd[:,50,30,:,:],[numX*numAng,numB]).transpose())
#    plt.show()
    
    
    return bvalNrrd, bvalues, Gnorm, scaleB, headerPath, uniqueGrads, valIX
    

#%%
def saveNRRDwithHeader( data, headerName, savePath, saveName , bvalues, gradients):
    
    
    dataDims = data.shape
    Bmax = np.max(bvalues[:])
    
    # write nrrd file with data
    saveNameSlice = range(dataDims[3])
    for ii in range(dataDims[3]):
        temp = data[:,:,:,ii]
        saveNameSlice[ii] =  saveName + '.Dir_' + str(ii+1).zfill(4) + '.nrrd'
        nrrd.write( savePath + saveNameSlice[ii], temp)
    
    # read in all data file
    dataFi = open(headerName,'r')
    dataLines = dataFi.readlines()
    dataFi.close()
    
    headerName = headerName.split('/')[-1]
    
    # reopen to write final maks file
    dataFi = open(savePath + saveName + '.nhdr','w')
    for ll in dataLines:
        
        if ll.find('type:') > -1:
            dataFi.writelines( 'type: float\n' )
        elif ll.find('line skip:') > -1:
            dataFi.writelines( 'line skip: 11\n' )
        elif ll.find('sizes:') > -1:
            dataFi.writelines( 'sizes: %d %d %d %d\n' %( dataDims ) )
        elif ll.find('DWMRI_b-value:') > -1:
            dataFi.writelines( 'DWMRI_b-value:=' + str(Bmax) + '\n' )
            break;
        else:
            dataFi.writelines( ll )
            
    for ii in range(dataDims[3]):
        normGradient = gradients[ii,:] * np.sqrt( bvalues[ii] / np.float(Bmax) )
        dataFi.writelines('DWMRI_gradient_' + str(ii).zfill(4)  + (':=%0.8f %0.8f %0.8f\n' %( normGradient[0], normGradient[1], normGradient[2] )))
        
    dataFi.writelines('data file: LIST\n')
    for ii in range(dataDims[3]):
        dataFi.writelines( saveNameSlice[ii] + '\n')
        
    dataFi.close()
    
    return 0

    