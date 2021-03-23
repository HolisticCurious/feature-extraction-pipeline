# -*- coding:utf-8 -*-

"""
Code modified on March 2021 by Maria Teresa Parreira @Instituto Superior Tecnico, Lisboa

"""

import numpy as np
from skimage.measure import regionprops, shannon_entropy
from scipy.stats import skew, kurtosis

class RegionPropsInt:
    """
    Various region properties (intensity-based)
    """

    def __init__(self, img, level_min=0, level_max=256):
        """
        initialize
        :param img: normalized image
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        """

        
        self.img = img
        self.bin_img = (img!=0)*1
        
        
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max

        hist,bin_edges = np.histogram(self.img.ravel(),level_max,[level_min,level_max])
        self.hist = np.array(hist)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        


        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values
        :return: feature values
        """
        
        
        features = {}
        
        props = regionprops(self.bin_img,self.img)

        features['Mean Intensity'] = np.mean(self.img)
        #mean intensity of image

        features['Std'] = np.std(self.img)
        #standard deviation

        features['Variance'] = np.var(self.img)
        #variance

        features['Skewness'] = skew(self.img, axis= None)
        #skewness of distribution

        features['Kurtosis'] = kurtosis(self.img, axis = None)
        #kurtosis of distribution

        features['Contrast'] = np.std(self.hist)
        #contrast can be defined as std of histogram of intensity
        
        features['Max Intensity'] = props[0].max_intensity
        #Value with the greatest intensity in the region.
        
        features['Min Intensity'] = props[0].min_intensity
        #Value with the greatest intensity in the region.
        
        features['Entropy'] = shannon_entropy(self.img, base=2)
        #The Shannon entropy is defined as S = -sum(pk * log(pk)), where pk are frequency/probability of pixels of value k.
        

        
        return features

    
    def print_features(self, print_values = True):
        """
        print features
        """
        
        if print_values:
            print("----RegionProps-----")
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])
            
        return feature_labels, feature_values
