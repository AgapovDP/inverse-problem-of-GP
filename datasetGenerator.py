# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Дима
"""


import numpy as np



randomObjectVector = np.random.randint(2, size = (1,4))

#Jones matrix of Linear Amplitude Anisotropy (LAA). 
def matrixLAA():
    M11 = 0
    M12 = 0
    M22 = 0
    return [[M11,M12],[M12,M22]]

#Jones matrix of Linear Phase Anisotropy (LPA).
def matrixLPA():
    M11 = 0
    M12 = 0
    M22 = 0
    return [[M11,M12],[M12,M22]]

#Jones matrix of Circular Amplitude Anisotropy (CAA)
def matrixCAA():
    M11 = 0
    M12 = 0
    M22 = 0
    return [[M11,M12],[M12,M22]]

#Jones matrix of Circular Phase Anisotropy (CPA)
def matrixCPA():
    M11 = 0
    M12 = 0
    M22 = 0
    return [[M11,M12],[M12,M22]]

def calculationFinalMatrix(objectVector):
    matrix = [[1,0],[0,1]]
    if objectVector[0] == 1: matrix = matrix.dot(matrixLAA())
    if objectVector[1] == 1: matrix = matrix.dot(matrixLPA())
    if objectVector[2] == 1: matrix = matrix.dot(matrixCAA())
    if objectVector[3] == 1: matrix = matrix.dot(matrixCPA())

def colcilationCorrFunctions():
    return 0



def main(lenDataset = 100):
    return 0