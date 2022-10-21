# -*-coding=utf-8-*-

# radiomics
from featureExtractor import feature_extract
from glob import glob
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm 

def z_score_normalize(train_dataset):
    mean = np.mean(train_dataset)
    std = np.std(train_dataset)
    #print ("train data mean, std", mean, std)
    return mean, std

def n4correction(input_img):
    '''
    :param input_img: numpy array format
    :return: n4 bias field corrected image with numpy array format
    '''
    
    inputImage = sitk.GetImageFromArray(input_img)
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    output = corrector.Execute(inputImage, maskImage)
    output = sitk.GetArrayFromImage(output)

    return output

save_dir2= "./preprocessed/"
filepath = "./IDHwt_Perfusion/*"
castFilter = sitk.CastImageFilter()
castFilter.SetOutputPixelType(sitk.sitkInt16)

all_images = []
mean = []
std = []
num = 0
for pat in tqdm(glob(filepath), total = len(glob(filepath))): 
    pat_name = pat.split("/")[-1]
    print(pat_name)
    try:
        for file in glob(pat +"/CT1fromhd/*"):
            filename = file.split("/")[-1]
            filetype = file.split("/")[-1].split(".nii")[0].split("_")[-1]
            img = sitk.ReadImage(file)[:,:,0]
            img_arr = sitk.GetArrayFromImage(img)

            image_np = img_arr.copy()
            mean1, std1 = z_score_normalize(image_np)
            mean.append(mean1)
            std.append(std1)
            num+=1

            image_np = (image_np - mean1)/std1

            n4 = n4correction(image_np)
            n4 = castFilter.Execute(n4)

            image_np_norm = sitk.GetImageFromArray(n4)
            image_np_norm.CopyInformation(img)
            if not os.path.exists(save_dir2 + pat_name):
                os.mkdir(save_dir2 + pat_name)
                
            sitk.WriteImage(image_np_norm, save_dir2 +"/"+ pat_name+"/"+filename)
    except:
        f = open('Null_Patients.txt','w')
        f.write(pat_name+'_CT1\n')

f.close()




# Radiomics Feature Extraction
import pandas as pd
import numpy as np
from glob import glob
import os
import SimpleITK as sitk
import radiomics.featureextractor
from tqdm import tqdm
import warnings 
import logging 
warnings.filterwarnings(action='ignore') 
radiomics.logger.setLevel(logging.ERROR)

def feature_extract(image_origin, image_mask, features = ['firstorder', 'glcm','gldm', 'glszm', 'glrlm', 'ngtdm', 'shape'], binWidth=None, binCount=None):
    '''
    :param image_origin: image_array (numpy array)
    :param image_mask: mask_array (numpy array)
    :return: whole features, featureVector
    '''
    
    image = image_origin
    mask = image_mask
    
    settings = {}

    if binWidth:
        settings['binWidth'] = binWidth
    if binCount:
        settings['binCount'] = binCount
    settings['resampledPixelSpacing'] = (1,1,1)
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.settings['enableCExtensions'] = True
    
    for feature in features:
        extractor.enableFeatureClassByName(feature.lower())
        
    featureVector = extractor.execute(image, mask)
    
    cols = []; feats = []
    for feature in features:
        for featureName in sorted(featureVector.keys()):
            if feature in featureName:
                cols.append(featureName)
                feats.append(featureVector[featureName])
    return feats, cols

def extract(id_df, image, mask, number):
#     feats, cols = feature_extract(image, mask, binWidth=number)
    feats, cols = feature_extract(image, mask, binCount=number)
    df = pd.DataFrame(feats, cols).T
    result = pd.concat([id_df, df], axis=1)
    
    return result

all_images = []
mean = []
std = []
num = 0

case = 'binCount'
# case = 'binWidth'
filepath = './preprocessed/*'

for pat in tqdm(glob(filepath),total=len(glob(filepath))): 
    pat_name = pat.split("/")[-1]
    for number in [8, 32, 128]:
        path = 'preprocessed/'+str(pat_name)+'/'
        if pat_name + "_T1GD_registered-label.nii" in os.listdir(path):
            mask = sitk.ReadImage(path + pat_name + "_T1GD_registered-label.nii")
        else :
            mask = sitk.ReadImage(path + pat_name + "_T2_registered-label.nii")

        T1GD_image = sitk.ReadImage(path+str(pat_name)+'_T1GD_registered.nii')
        T2_image = sitk.ReadImage(path+str(pat_name)+'_T2_registered.nii')
        normalize_T1GD_image = sitk.ReadImage('normalization/'+str(pat_name)+'/'+str(pat_name)+'_T1GD_registered.nii')
        normalize_T2_image = sitk.ReadImage('normalization/'+str(pat_name)+'/'+str(pat_name)+'_T2_registered.nii')

        id_df = pd.DataFrame([pat_name], ['ID']).T

        T1GD_result = extract(id_df, T1GD_image, mask, number)
        T2_result = extract(id_df, T2_image, mask, number)
        normalize_T1GD_result = extract(id_df, normalize_T1GD_image, mask, number)
        normalize_T2_result = extract(id_df, normalize_T2_image, mask, number)

        if pat_name == glob(filepath)[0].split('/')[-1]:
            T1GD_result.to_csv('results/'+case+'/T1GD_'+case+'_'+str(number)+'.csv', mode='w', header=True, index=False)
            T2_result.to_csv('results/'+case+'/T2_'+case+'_'+str(number)+'.csv', mode='w', header=True, index=False)
            normalize_T1GD_result.to_csv('results/'+case+'/normalize_T1GD_'+case+'_'+str(number)+'.csv', mode='w', header=True, index=False)
            normalize_T2_result.to_csv('results/'+case+'/normalize_T2_'+case+'_'+str(number)+'.csv', mode='w', header=True, index=False)
        else:
            T1GD_result.to_csv('results/'+case+'/T1GD_'+case+'_'+str(number)+'.csv', mode='a', header=False, index=False)
            T2_result.to_csv('results/'+case+'/T2_'+case+'_'+str(number)+'.csv', mode='a', header=False, index=False)
            normalize_T1GD_result.to_csv('results/'+case+'/normalize_T1GD_'+case+'_'+str(number)+'.csv', mode='a', header=False, index=False)
            normalize_T2_result.to_csv('results/'+case+'/normalize_T2_'+case+'_'+str(number)+'.csv', mode='a', header=False, index=False)

