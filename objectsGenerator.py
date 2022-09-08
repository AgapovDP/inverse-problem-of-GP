# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""
# Jones matrix ----- DOI: 10.1103/PhysRevE.74.056607

import numpy as np
import PolarizationObject
         
def save_Object(path, lenDataset):
    dataset = []
    for i in range(lenDataset):
        polObject = PolarizationObject.PolarizationObject()
        polObject.change_properties()
        polObject.calculation_Corr_Functions()
        # теперь осталось чтоб он сохранял только не нан параметры! 
        parameters = np.array([],np.single)
        
        for j in polObject.setOfParametrs:
            if np.isnan(j) == False: parameters = np.append(parameters,j)
        
        dataset.append(([polObject.setOfCorrFunc,\
                           polObject.classVector, parameters]))
    np.save(path,dataset)
        
                 
if __name__ == "__main__":
    save_Object("datasetExample",10)
    