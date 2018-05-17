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
def pullTSCDatasets( inputPath, savePath, ageTarget=np.NaN, genderTarget='' ,targetFolder='diffusion/cusp90/01-motioncorrection/' , targetHeader='_mocoraff_diffusion.nhdr'):
    
    # define
    caseRef = 'case'
    processedPath = inputPath + 'Processed/'
    rawPath = inputPath + 'RAW/'
    
    # prealocate
    subjectsData = []
    subjID = []
    scanID = []
    
    # determine valid cases
    inputFolders = os.listdir(processedPath)
    for fol in inputFolders:
        
        # if valid reference
        if fol.lower().find(caseRef) > -1 :
            
            try: 
                
                print 'Retrieve: ' + fol
                
                # retrieve subject ID and subFoldName
                subjID.append(int(fol.lower().split(caseRef)[-1]))
                subFoldName = os.listdir( processedPath + fol )[0]
                scanFold = os.listdir( processedPath + fol +'/' +subFoldName+'/')
                
                # gert available runs
                
                scanID.append([])
                for ss in scanFold:
                    if ss.find('scan') > -1:
                        
                        # generate load path
                        baseFolder = processedPath + fol +'/' +subFoldName+'/' + ss + '/common-processed/'
                        loadFolder = processedPath + fol +'/' +subFoldName+'/' + ss + '/common-processed/' + targetFolder 
                        
                        # retrieve subject info
                        fi = open( rawPath + fol +'/' +subFoldName+'/' + ss + '/data_for_analysis/scaninfo.csv')
                        lines = fi.readlines()
                        fi.close()
                        age = float(lines[1].split(',')[np.where([ 'age' == s for s in lines[0].split(',') ])[0][0]])
                        gender = lines[1].split(',')[np.where([ 'gender' == s for s in lines[0].split(',') ])[0][0]]
                        
                        
                        if ( (ageTarget - 2 < int(age) < ageTarget + 2) | np.isnan(ageTarget) ) &\
                        ( (  gender == genderTarget ) | ( genderTarget == '' ) )  &\
                        os.path.exists( loadFolder ):
                            
                            scanID[-1].append(  int(ss.split('scan')[-1]) )
                            
                            # subj id
                            subjSTR = 'c' + str(subjID[-1]).zfill(3) +'_s' + str(scanID[-1][-1]).zfill(2) 
                            
                            # diff header
                            diffusionHeader = subjSTR + targetHeader
                            
                            # get mask path
                            maskPath =  subjSTR + '_uncorrected_dwi_icc.nrrd'  
                            
                            # generate load path
                            saveFolder=  savePath + fol +'/' +subFoldName+'/' + ss + '/commmon-processed/' 
                            
                            # save data
                            if (len(subjectsData) == 0):
                                subjectsData.append({ 'subjID': subjID[-1], 'dataPaths' : []})
                            elif (subjectsData[-1]['subjID'] != subjID[-1]):
                                subjectsData.append({ 'subjID': subjID[-1], 'dataPaths' : []})
                                
                            subjectsData[-1]['dataPaths'].append( { 'subjSTR':subjSTR, 'baseFolder': baseFolder, \
                                                                    'loadFolder' : loadFolder, 'diffHeader' : diffusionHeader, \
                                                                    'maskPath' : maskPath, 'saveFolder' : saveFolder, 'subjID' : subjID[-1], \
                                                                    'scanID': scanID[-1][-1], 'age':age, 'gender':gender } )
                
            except:
                print '%s  is not a valid case' %( fol )
    
    
    return subjectsData


    
    