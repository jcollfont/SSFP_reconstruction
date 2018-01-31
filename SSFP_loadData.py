#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:02:47 2018

@author: jaume
"""

## LOAD LIBRARIES
import numpy as np
import nrrd

## SSFP LOAD
#		This file loads the SSFP files and the T1 and T2 images from the given paths
#
#
def loadSSFPdata( loadSSFPpaths, loadT1path, loadT2path ):

    ## Load T1
    T1_img = np.array(nrrd.read( loadT1path )[0])

    ## Load T2
    T2_img = np.array(nrrd.read( loadT2path )[0])

    ## Load SSFP
    numSSFP = len(loadSSFPpaths)
    SSFP_files = range(numSSFP)
    SSFP_shape = range(numSSFP)
    for ss in range(numSSFP):
        print('Loading file: ' + loadSSFPpaths[ss] )
        SSFP_files[ss] = np.rray( nrrd.read( loadSSFPpaths[ss] )[0] )
        SSFP_shape = SSFP_files[ss].shape

    return SSFP_files, T1_img, T2_img

## LOAD gradients
#		This function loads the gradients from the ones provided in the given file.
#		The format of the file must be:
#
#		E.g.
#			[directions=3]
#			coordinatesystem[0]=xyz
#			normalisation[0]=unity
#			vector[0]=(  1.0,0.0,0.0 )
#			vector[1]=(  0.0,1.0,0.0 )
#			vector[2]=(  0.0,0.0,1.0 )
#
#
def loadGradients( gradInfoPath , scaleB):

	# load file
	hddr = open( gradInfoPath ,'r')
	lines = hddr.readlines()
	hddr.close()

	# extract direction groups
	directionGroups = []
	for li in range(len(lines)):
		if (lines[li].find("directions=") > -1):
			directionGroups.append( {'line' : li , 'numDir' : int(lines[li].split("=")[1][0:-3]) , 'directions' :[] , 'normGrad' : [] , 'bvalues' : []} )

			# extract direction
			for jj in range(directionGroups[-1]['numDir']):
				directionGroups[-1]['directions'].append( eval('[' + lines[ li + 3 +jj ].split('=')[1] + ']') )

	# Compute B values and normalize gradients
	for ii in range(len(directionGroups)):
		directionGroups[ii]['bvalues'] = np.ndarray([ directionGroups[ii]['numDir'],1 ])
		directionGroups[ii]['normGrad']  = range( directionGroups[ii]['numDir'] )

		for jj in range(directionGroups[ii]['numDir']):
			normGrad = np.linalg.norm(directionGroups[ii]['directions'][jj],2)
			directionGroups[ii]['bvalues'][jj] = scaleB*pow(normGrad,2)
			directionGroups[ii]['normGrad'][jj] = directionGroups[ii]['directions'][jj]/normGrad

	return directionGroups
