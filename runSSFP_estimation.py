#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:39:28 2018

@author: ch199899
"""
#%% IMPORTS
import numpy as np
import nrrd
from loadDWIdata import loadDWIdata
from SSFP_functions import computeAllSSFPParams


#%% set paths
loadFolder = 'data_DWI/CSR_2/'
ssfpFolder = 'common-processed/diffusion/cusp90/01-motioncorrected/'
header = 'something.nhdr'
t1wPath = 'common-processed/diffusion/cusp90/03-dwi2t1w_registration/c035_s02_t1w-to-b0.nrrd'
t2wPath = 'common-processed/diffusion/cusp90/03-dwi2t1w_registration/c035_s02_t2w-to-b0.nrrd'

#%% set params
alpha = 30
M0 = 3
N = 5
TR = 2

#%% load data
dataSSFP, bvalues, Gnorm, scaleB, headerPath, uniqueGrads, valIX = loadDWIdata(loadFolder, header)

t1wimg = nrrd.read(t1wPath)[0]
t2wimg = nrrd.read(t2wPath)[0]

#%% Compute params
K, L, E1, E2 = computeAllSSFPParams(t1wimg, t2wimg, TR, alpha, M0, N)
