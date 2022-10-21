from __future__ import print_function
import collections
# from radiomics.base import RadiomicsFeaturesBase
import logging
from radiomics import featureextractor
import radiomics
import logging
radiomics.logger.setLevel(logging.ERROR)
import SimpleITK as sitk
from radiomics import firstorder, glcm, shape, glrlm, glszm, ngtdm, gldm
import numpy as np


#, 'glcm', 'glszm', 'glrlm', 'ngtdm', 'shape'
def Numpy2Itk(array):
    '''
    :param array: numpy array format
    :return: simple itk image type format
    '''
    return sitk.GetImageFromArray(array)

def feature_extract(image_origin, image_mask, features = ['firstorder', 'glcm', 'glszm', 'glrlm', 'ngtdm', 'shape'], imagetypes=['Original','LoG','Wavelet']):
    '''
    :param image_origin: image_array (numpy array)
    :param image_mask: mask_array (numpy array)
    :subject: subject name
    :return: whole features, featureVector, make csv_file
    '''
    # print(image_origin.shape)
    # print(image_mask.shape)
    
    image = Numpy2Itk(image_origin)
    mask = Numpy2Itk(image_mask)

    settings = {}
    
    if binWidth:
        settings['binWidth'] = binWidth

    if binCount:
        settings['binCount'] = binCount
    # settings['binWidth'] = 8
    # settings['binCount'] = 128
#     settings['resampledPixelSpacing'] = (1,1,1)
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    settings['sigma'] = [0.5,1.5,2.5]
#     settings['normalize'] = True
#     settings['normalizeScale'] = 100
#     settings['removeOutliers'] = False
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.settings['enableCExtensions'] = True
  
    for feature in features:
        extractor.enableFeatureClassByName(feature.lower())
    #image preprocessing customizing fashion
    # for imagetype in imagetypes:
    #     extractor.enableImageTypeByName(imagetype,enabled=True, customArgs = None)
    extractor.enableAllImageTypes()
  
    featureVector = extractor.execute(image, mask)
#     print(featureVector.keys())
    

    cols = []; feats = []
    for feature in features:
        for featureName in sorted(featureVector.keys()):
            if feature in featureName:
                cols.append(featureName)
                feats.append(featureVector[featureName])
                
    return feats, cols
