# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""

import datasetGenerator


#длина первой группы данных
lenFirstGroup = 100000
NOISE_1 = 0.0
NOISE_2 = 0.01
NOISE_3 = 0.05
NOISE_4 = 0.1

#datasetGenerator.dataGeneratorRandomObject("datasets/datasetNoNoise_randomObject_train_"+str(lenFirstGroup),lenFirstGroup,NOISE_1)
#datasetGenerator.dataGeneratorRandomObject("datasets/datasetNoNoise_randomObject_test_"+str(lenFirstGroup),lenFirstGroup,NOISE_1)
#datasetGenerator.dataGeneratorLAA("datasets/datasetNoNoise_LAA_train_"+str(lenFirstGroup),lenFirstGroup,NOISE_1)
#datasetGenerator.dataGeneratorLAA("datasets/datasetNoNoise_LAA_test_"+str(lenFirstGroup),lenFirstGroup,NOISE_1)
print("Step 2")
#datasetGenerator.dataGeneratorRandomObject("datasets/datasetLowNoise_randomObject_train_"+str(lenFirstGroup),lenFirstGroup,NOISE_2)
#datasetGenerator.dataGeneratorRandomObject("datasets/datasetLowNoise_randomObject_test_"+str(lenFirstGroup),lenFirstGroup,NOISE_2)
#datasetGenerator.dataGeneratorLAA("datasets/datasetLowNoise_LAA_train_"+str(lenFirstGroup),lenFirstGroup,NOISE_2)
#datasetGenerator.dataGeneratorLAA("datasets/datasetLowNoise_LAA_test_"+str(lenFirstGroup),lenFirstGroup,NOISE_2)
print("Step 3")
#datasetGenerator.dataGeneratorRandomObject("datasets/datasetMediumNoise_randomObject_train_"+str(lenFirstGroup),lenFirstGroup,NOISE_3)
#datasetGenerator.dataGeneratorRandomObject("datasets/datasetMediumNoise_randomObject_test_"+str(lenFirstGroup),lenFirstGroup,NOISE_3)
#datasetGenerator.dataGeneratorLAA("datasets/datasetMediumNoise_LAA_train_"+str(lenFirstGroup),lenFirstGroup,NOISE_3)
#datasetGenerator.dataGeneratorLAA("datasets/datasetMediumNoise_LAA_test_"+str(lenFirstGroup),lenFirstGroup,NOISE_3)
print("Step 3")
datasetGenerator.dataGeneratorLPA("datasets/datasetLargeNoise_LPA_train_"+str(lenFirstGroup)+'_V1',lenFirstGroup,NOISE_4)
datasetGenerator.dataGeneratorLPA("datasets/datasetLargeNoise_LPA_test_"+str(lenFirstGroup)+'_V1',lenFirstGroup,NOISE_4)