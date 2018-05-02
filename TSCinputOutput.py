#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:49:40 2018

@author: ch199899
"""
#%% IMPORTS
from loadDWIdata import loadDWIdata
import numpy as np
import nrrd
import os



#%% TSC params

groupFolder = { 'tsc': 'TSC/', 'autism':'Autism/', 'control':'Controls/' } 


    
#%% pull TSC data
def pullTSCDatasets( inputPath, savePath ):
    
    # define
    caseRef = 'case'
    
    # prealocate
    subjectsData = []
    subjID = []
    scanID = []
    
    # determine valid cases
    inputFolders = os.listdir(inputPath)
    for fol in inputFolders:
        
        # if valid reference
        if fol.lower().find(caseRef) > -1 :
            
            try: 
                
                print 'Retrieve: ' + fol
                
                # retrieve subject ID and subFoldName
                subjID.append(int(fol.lower().split(caseRef)[-1]))
                subFoldName = os.listdir( inputPath + fol )[0]
                scanFold = os.listdir( inputPath + fol +'/' +subFoldName+'/')
                
                # gert available runs
                subjectsData.append({ 'subjID': subjID[-1], 'dataPaths' : []})
                scanID.append([])
                for ss in scanFold:
                    if ss.find('scan') > -1:
                        
                        # generate load path
                        loadFolder = inputPath + fol +'/' +subFoldName+'/' + ss + '/common-processed/diffusion/cusp90/01-motioncorrection/' 
                        
                        if os.path.exists( loadFolder ):
                            
                            scanID[-1].append(  int(ss.split('scan')[-1]) )
                            
                            # subj id
                            subjSTR = 'c' + str(subjID[-1]).zfill(3) +'_s' + str(scanID[-1][-1]).zfill(2) 
                            
                            # diff header
                            diffusionHeader = subjSTR + '_mocoraff_diffusion.nhdr'
                            
                            # get mask path
                            maskPath =  subjSTR + '_uncorrected_dwi_icc.nrrd'  
                            
                            # generate load path
                            saveFolder=  savePath + fol +'/' +subFoldName+'/' + ss + '/commmon-processed/' 
                    
            
                            subjectsData[-1]['dataPaths'].append( { 'loadFolder' : loadFolder, 'diffHeader' : diffusionHeader, 'maskPath' : maskPath, 'saveFolder' : saveFolder, 'subjID' : subjID[-1], 'scanID': scanID[-1][-1] } )
                
            except:
                print '%s  is not a case' %( fol )
    
    
    return subjectsData