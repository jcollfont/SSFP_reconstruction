#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:26:58 2018

@author: Jaume Coll-Font
"""
## HELP:
#       This set of functions implements the reconstruction of the diffusion
#       weighted decay obtained from Steady State Free Precession (SSFP) sequences.
#       Details about this implementation can be found in the paper:
#
#       J. A. McNab and K. L. Miller,  “Sensitivity of diffusion weighted
#       steady state free precession to anisotropic diffusion,”
#       Magn. Reson. Med., vol. 60, no. 2, pp. 405–413, 2008.
#
#
#
#


## IMPORTS
import numpy as np


## SSFP fit
#       This function resolves the component of weighted diffusion from the
#       SSFP sequences and the precomputed weighted decays of T1 and T2.
#
#
#       INPUT:
#           - M0 - double - nominal magnetic field strength
#           - E1 - <X,Y,Z>double - Longitudonal magnetization
#           - E2 - <X,Y,Z>double - Tranverse magnetization
#			- alpha - [0,2*pi] double - flip angle of RF pulse
#			- S - <X,Y,Z,N>double - Reconstructed SSFP images
#
#		OUTPUT:
#			- E_diff - <X,Y,Z,N>double - Reconstructed diffusion weighting from
#										SSFP images.
#
#
def SSFP_fit( M0, E1, E2, alpha, S):

    ## general params
    N = len(S)

    ## Compute K
    K = -M0*(1-E1)*E1*E2**2*np.sin(alpha) / ( 1 - E1*np.cos(alpha) )

    ## Compute L
    L = range(N)
    E_diff = range(N)
    for n in range(N):
        L[n] = computeL(E1, alpha, n)

        E_diff[n] = S[n]/K/L[n]

    return E_diff

## Compute L
#	Subfunction of SSFP_fit.
#
def computeL( E1, alpha, n):

    if n == 0:
        Ln = E1**(-1)*(1-np.cos(alpha))
    elif n > 0:
        Ln = np.sin(alpha)**2*(E1*np.cos(alpha))**(n-1)

    return Ln

## Compute E1
#		This function is necessary to compute the T1 weighted images.
#		It extracts the T1 time of each voxel and returns the
#		longitudonal magnetization after one repetition period
#
#		INPUT:
#			- img - <X,Y,Z,N>double - T1 weighted images
#			- TR - double  -  Time of repetition
#
#		OUTPUT:
#			- E1 - <X,Y,Z>double - Longitudonal magnetization
#
def computeE1(img, TR):

	## Extract T1 params
	T1 = 1

	## Compute E1
	E1 = np.exp(-TR*1/T1)

	return E1

## Compute E2
#		This function is necessary to compute the T1 weighted images.
#		It extracts the T2 time of each voxel and returns the
#		transverse magnetization after one repetition period
#
#		INPUT:
#			- img - <X,Y,Z,N>double - T2 weighted images
#			- TR - double  -  Time of repetition
#
#		OUTPUT:
#			- E2 - <X,Y,Z>double - Longitudonal magnetization
#
def computeE1(img, TR):

	## Extract T2 params
	T2 = 2

	## Compute E2
	E2 = np.exp(-TR*1/T2)

	return E2
