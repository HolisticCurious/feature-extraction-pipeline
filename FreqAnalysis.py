# -*- coding:utf-8 -*-
"""
Code modified on March 2021 by Maria Teresa Parreira @Instituto Superior Tecnico, Lisboa


"""
import numpy as np
from scipy import ndimage as ndi


from skimage.filters import gabor_kernel
from skimage.measure import shannon_entropy
from scipy import signal
from statistics import mean 


class FreqAnalysis:
    """
    Analysing data from frequency standpoint
    """

    def __init__(self, img):
        """
        initialize
        :param img: normalized image

        """

        features = {}
        self.img = img
        
        
        # prepare filter bank kernels
        kernels = []
        powers = []
        frequencies = np.array([0.1, 0.2,0.3,0.4]) #4 frequencies
        for theta in range(4):
            theta = theta / 4. * np.pi #4 angles
            for frequency in frequencies: #a total of 16 filters
                kernel_temp = gabor_kernel(frequency, theta = theta)
                power_img = self.power_calc(img,kernel_temp)
                powers.append (np.mean(power_img))
                #save only real kernel
                kernels.append(np.real(kernel_temp))
          
        features['Mean Gabor Power'] = mean(powers)
                    
        # prepare reference features

        feat_gabor = self.compute_gabor_feats(self.img, kernels)
        
        features.update(feat_gabor)
        
        feat_freq = self.compute_freq_feats(self.img)
        
        features.update(feat_freq)
        
        self.features = features


    def compute_gabor_feats(self,image, kernels):
        
        features = {}
        f_var = []
        f_mean = []
        f_energy = []
        f_ent = []
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            f_mean.append(np.mean(filtered))
            f_var.append(np.var(filtered))
            f_energy.append(np.sum(np.power(filtered.ravel(),2))/len(filtered.ravel()))
            f_ent.append(shannon_entropy(filtered))
        
        features['Gabor Variance'] = mean(f_var)
        features['Gabor Mean'] = mean(f_mean)
        features['Gabor Energy'] = mean(f_energy)
        features['Gabor Entropy'] = mean(f_ent)
        
        
        return features
        
        
        
        
    def compute_freq_feats(self,image):
        #get some spectral analysis features

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        
        features = {}
        features['Mean Spectral Magnitude'] = np.mean(magnitude_spectrum) #in dB
        
        f_psd, Pxx = signal.welch(image)
        
        sum_fp = np.multiply(f_psd, Pxx)
        sum_int = np.sum(Pxx)
        
        mean_pxx = np.sum(sum_fp)/sum_int
        
        features['Mean Spectral Power'] = mean_pxx
        
        return features
           
        
        


    def power_calc(self,image,kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        #convolves images with filter
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)





    
    def print_features(self, print_values = True):
        """
        print features
        """
        
        if print_values:
            print("----Frequency Analysis-----")
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])
            
        return feature_labels, feature_values

