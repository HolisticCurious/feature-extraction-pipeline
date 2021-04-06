# -*- coding:utf-8 -*-

"""
Code modified on March 2021 by Maria Teresa Parreira @Instituto Superior Tecnico, Lisboa

"""

import numpy as np
import math
from skimage.measure import regionprops
from mahotas.features import zernike_moments




class RegionPropsMorph:
    """
    Various region properties (morphological)
    """

    def __init__(self, img):
        """
        initialize
        :param img: normalized image
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        """

        
        self.img = img
        self.bin_img = (img!=0)*1
        

        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values
        :return: feature values
        """
        
        
        features = {}
        
        props = regionprops(self.bin_img,self.img)
        
        features['Area'] = props[0].area #Number of pixels of the region.
        
        features['BB Area'] = props[0].bbox_area #Number of pixels of bounding box.

        features['Perimeter'] = props[0].perimeter
        #Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
        
        features['Centroid'] = props[0].centroid #Centroid coordinate tuple.
        
        features['Weighted Centroid'] = props[0].weighted_centroid 
        #Centroid coordinate tuple (row, col) weighted with intensity image.
        
        features['Centroid Divergence'] = np.linalg.norm(np.array(props[0].centroid) - np.array(props[0].weighted_centroid))
        
        features['Equivalent Diameter'] = props[0].equivalent_diameter
        #The diameter of a circle with the same area as the region.
        
        features['Major Axis Length'] = props[0].major_axis_length 
        #The length of the major axis of the ellipse that has the same normalized second central moments as the region.
        
        features['Minor Axis Length'] = props[0].minor_axis_length 
        #The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
        
        features['Eccentricity'] = props[0].eccentricity 
        #Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points).
        
        features['Circularity'] = (4*props[0].area*math.pi)/(props[0].perimeter**2)
        #Circularity that specifies the roundness of objects.

        features['Roundness'] = (4*props[0].area)/(np.pi* props[0].major_axis_length**2 )
        #Like circularity, but does not depend on perimeter/roughness.

        features['Aspect Ratio'] = (props[0].major_axis_length)/ props[0].minor_axis_length
        #Aspect ratio.
        
        features['Orientation'] = props[0].orientation
        #Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

        features['Solidity'] = props[0].solidity
        #Ratio of pixels in the region to pixels of the convex hull image.

        conv_img = props[0].convex_image #get convex image of ROI
        conv_perimeter =  regionprops(conv_img.astype(np.uint8))[0].perimeter

        features['Roughness'] = props[0].perimeter/conv_perimeter
        #Ratio of perimeter of region to perimeter of the convex hull image.

        features['Hu Moments'] = (props[0].weighted_moments_hu)
        #tuple - Hu moments (translation, scale and rotation invariant) of intensity image.
        
        diam = self.img[0].shape[0]

        maxradius = diam/2
        features['Zernike Moments'] = zernike_moments(self.img,maxradius)
        #Zernike Moments of Region





        return features

    
    def print_features(self, print_values = True):
        """
        print features
        """
        
        if print_values:
            print("----RegionPropsMorph-----")
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])
            
        return feature_labels, feature_values
