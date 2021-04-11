# -*- coding:utf-8 -*-
"""
Code modified on March 2021 by Maria Teresa Parreira @Instituto Superior Tecnico, Lisboa

    GLCM
    Copyright (c) 2016 Tetsuya Shinaji
    This software is released under the MIT License.
    http://opensource.org/licenses/mit-license.php

    Date: 2016/01/29
"""

import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy



class GLCM:
    """
    Gray-Level Co-occurrence Matrix
    """

    def __init__(self, img, level_min=1, level_max=256, threshold=None):
        """
        initialize
        :param img: normalized image
        :param theta: definition of neighbor
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        :param threshold: threshold of the minimal value
        """

        
        self.img = img
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max
        #self.theta = theta
        self.glcm = greycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed = True)
        #The grey-level co-occurrence histogram. The value P[i,j,d,theta] is the number of times that grey-level j 
        #occurs at a distance d and at an angle theta from grey-level i. If normed is False, the output is of type 
        #uint32, otherwise it is float64. The dimensions are: levels x levels x number of distances x number of angles.
        
        #here, we have 4 matrices

        matrix = np.sum(self.glcm, axis = 3) / 4 #makes the matrix invariant as it sums all the elements for different angles
        self.matrix = np.ndarray((256,256,1,1))#keeps it 4-dimensional
        self.matrix[:,:,:,0] = matrix


        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values
        :return: feature values
        """
        
        
        features = {}
        unif = []
        ent = []
        for i in np.arange(self.glcm.shape[3]):
            mat = self.glcm[:,:,0,i]
            feature_unif = (mat ** 2).sum()
            unif.append(feature_unif)
            feature_ent = shannon_entropy(mat)
            ent.append(feature_ent)
            
        matrix = self.matrix[:,:,0,0]
        features['Uniformity'] = list(unif) #uniformity for each matrix/angle
        features['Invariant Uniformity'] = (matrix ** 2).sum()
        
        features['GLCM Entropy'] = list(ent) #same, but for entropy
        features['GLCM Invariant Entropy'] = shannon_entropy(matrix)
        
        features['Correlation'] = greycoprops(self.glcm, 'correlation')[0]
        aux_corr = greycoprops(self.matrix, 'correlation')
        features['Invariant Correlation'] = float(aux_corr[0][0])
        
        features['Dissimilarity'] = greycoprops(self.glcm, 'dissimilarity')[0]
        aux_diss = greycoprops(self.matrix, 'dissimilarity')
        features['Invariant Dissimilarity'] = float(aux_diss[0][0])
                                                
        features['Contrast'] = greycoprops(self.glcm, 'contrast')[0]
        aux_cont = greycoprops(self.matrix, 'contrast')
        features['Invariant Contrast'] = float(aux_cont[0][0])
        
        features['Homogeneity'] = greycoprops(self.glcm, 'homogeneity')[0]
        aux_hom = greycoprops(self.matrix, 'homogeneity')
        features['Invariant Homogeneity'] = float(aux_hom[0][0])
        
        features['Energy'] = greycoprops(self.glcm, 'energy')[0]
        aux_eng = greycoprops(self.matrix, 'energy')
        features['Invariant Energy'] =  float(aux_eng[0][0])
        
        return features

    
    
    def print_features(self, print_values = True):
        """
        print features
        """
        
        if print_values:
            print("----GLCM-----")
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])
            
        return feature_labels, feature_values

