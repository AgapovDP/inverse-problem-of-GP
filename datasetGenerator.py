# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Дима
"""
# матрицы Джонса определены в соответствии со статьей DOI: 10.1103/PhysRevE.74.056607

import numpy as np
import random 

# тестовый вектор
randomObjectVector = np.random.randint(2, size = (1,4))

#Jones matrix of Linear Amplitude Anisotropy (LAA). 
def matrixLAA(theta = 0, P = 1, randomMod = True):
    if randomMod: 
        P = random.uniform(0,1)
        theta = random.uniform(-np.pi/2,np.pi/2,)
    M11 = np.cos(theta)**2 + P*np.sin(theta)**2
    M12 = M21 = (1-P)*np.cos(theta)*np.sin(theta)
    M22 = np.sin(theta)**2 + P*np.cos(theta)**2
    return np.array([[M11,M12],[M21,M22]])

#Jones matrix of Linear Phase Anisotropy (LPA).
def matrixLPA(alpha = 0, delta = 0, randomMod = True):
    if randomMod: 
        delta = random.uniform(0,2*np.pi)
        alpha = random.uniform(-np.pi/2,np.pi/2,)
    M11 = np.cos(alpha)**2 + np.exp(-1j*delta)*np.sin(alpha)**2
    M12 = M21 = (1-np.exp(-1j*delta))*np.cos(alpha)*np.sin(alpha)
    M22 = np.sin(alpha)**2 + np.exp(-1j*delta)*np.cos(alpha)**2
    return np.array([[M11,M12],[M21,M22]])

#Jones matrix of Circular Amplitude Anisotropy (CAA)
def matrixCAA(R = 0,randomMod = True):
    if randomMod: R = random.uniform(-1,1)
    M11 = M22 = 1.
    M12 = -1j*R
    M21 = 1j*R
    return np.array([[M11,M12],[M21,M22]])

#Jones matrix of Circular Phase Anisotropy (CPA)
def matrixCPA(phi = 0,randomMod = True):
    if randomMod: phi = random.uniform(0,2*np.Pi)
    M11 =  M22 = np.cos(phi)
    M12 = np.sin(phi)
    M21 = -np.sin(phi)
    return np.array([[M11,M12],[M21,M22]])

#function to calculete final Jones matrix
def calculationFinalMatrix(objectVector):
    matrix = np.array([[1,0],[0,1]])
    if objectVector[0] == 1: matrix = matrix.dot(matrixLAA())
    if objectVector[1] == 1: matrix = matrix.dot(matrixLPA())
    if objectVector[2] == 1: matrix = matrix.dot(matrixCAA())
    if objectVector[3] == 1: matrix = matrix.dot(matrixCPA())
    return matrix

def colcilationCorrFunctions():
    return 0



def main(lenDataset = 100):
    return 0