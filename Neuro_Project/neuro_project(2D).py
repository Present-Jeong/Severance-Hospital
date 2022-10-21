# -*-coding=utf-8-*-

# radiomics
from featureExtractor import feature_extract

# file_path
from glob import glob
import os

# read image
# import pydicom as dcm
import SimpleITK as sitk

# save as table
import pandas as pd

# mT
import numpy as np
import matplotlib.pyplot as plt

# PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# K-means
#from sklearn.cluster import KMeansnon_STD

# MDS
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

import random
from tqdm import tqdm 

#collage radiomics
# import collageradiomics

# Pixel Resampling
# Resampling

def resample(sitk_file, new_spacing = (1, 1, 1)):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_file.GetDirection())
    resample.SetOutputOrigin(sitk_file.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    orig_size = np.array(sitk_file.GetSize(), dtype=np.int)
    orig_spacing = sitk_file.GetSpacing()

    orig_spacing = np.array(orig_spacing)
    new_spacing = np.array(new_spacing)

    
    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    return resample.Execute(sitk_file)

filepath = "./Radiomics_dataset/*"
mask_path = "./normalization/label/"
save_path= "./preprocessed/"

# for pat in tqdm(glob(filepath), total = len(glob(filepath))): 
#     pat_name = pat.split("/")[-1]
#     # print(pat_name)
#     # print(glob(pat+'/*'))
#     for file in glob(pat+'/*'):
#         filename = file.split("/")[-1].split(".nii")[0]
#         print(filename)
#         img = sitk.ReadImage(file)
        
#         print("  Spacing info: ",img.GetSpacing())
#         print("  Size info: ", img.GetSize())
#         print("  Direction: ", img.GetDirection())
#         print("  Origin cord: ", img.GetOrigin())

#         # # Resample
#         print("**respacing img**")
#         resampledImg = resample(img)
#         print("  Image Spacing info: ", resampledImg.GetSpacing())
#         print("  Image Size info: ", resampledImg.GetSize())

#         if not os.path.exists(save_path + pat_name):
#             os.mkdir(save_path + pat_name)

#         sitk.WriteImage(resampledImg, save_path + pat_name + "/" + filename + ".nii")




# z-score Normalization

# filepath = os.listdir("./Radiomics_dataset/")

# def z_score_normalize(train_dataset):
#     mean = np.mean(train_dataset)
#     std = np.std(train_dataset)
#     #print ("train data mean, std", mean, std)
#     return mean, std

# filepath = "./preprocessed/*"
# save_dir= "./normalization/"

# all_images = []
# mean = []
# std = []
# num = 0
# for pat in tqdm(glob(filepath), total = len(glob(filepath))): 
#     pat_name = pat.split("/")[-1]
#     # print("Patient: ", pat_name)
#     for file in glob(pat + "/*"):
#         file_name = file.split('/')[-1].split('.nii')[0]
#         image_nii = sitk.ReadImage(file)
#         image_np = sitk.GetArrayFromImage(image_nii)
#         mean1, std1 = z_score_normalize(image_np)
#         mean.append(mean1)
#         std.append(std1)
#         num+=1

#         if 'label' not in file:
#             image_np_norm = (image_np - mean1)/std1
#         else :
#             image_np_norm = image_np

#         img_save_dir = os.path.join('normalization', 'image')
#         lab_save_dir = os.path.join('normalization','label')
#         if not os.path.exists(img_save_dir):
#             os.makedirs(img_save_dir)
#         if not os.path.exists(lab_save_dir):
#             os.makedirs(lab_save_dir)

#         if 'label' in file:
#             lab_file_path = save_dir +  "label/" + file_name + ".npy"
#             np.save(lab_file_path, image_np_norm)
#         else :
#             img_file_path = save_dir +  "image/" + file_name + ".npy"
#             np.save(img_file_path, image_np_norm)

#         print(file_name +':',np.max(image_np_norm),np.min(image_np_norm))

# mean = np.sum(mean) / num # useless
# std = np.sum(std) / num

# print(num)
# print(mean)
# print(std)

# Feature Extract
filepath = "./normalization/image/*"
mask_path = "./normalization/label/"
cnt = 0
res = ""
First = True

error_list = []

for pat in tqdm(glob(filepath), total = len(glob(filepath))):
    pf_name = pat.split("/")[-1].split("_")[0]
    image_np = np.load(pat)
    file_name = pat.split("/")[-1].split(".npy")[0].split("_")[1]
    if pf_name + "_T1GD_registered-label.npy" in os.listdir(mask_path):
        mask_manual_np = np.load(mask_path + pf_name + "_T1GD_registered-label.npy")
    else :
        mask_manual_np = np.load(mask_path + pf_name + "_T2_registered-label.npy")
    mask_manual_np = np.where(mask_manual_np>0, 1, mask_manual_np)

    # 1band to 3 bands
    image_np = np.stack((image_np,)*3,axis=-1)
    mask_manual_np = np.stack((mask_manual_np,)*3,axis=-1)
    # print(image_np.shape)
    # print(mask_manual_np.shape)
    # plt.imshow(image_np[:,:,0], cmap = 'gray')
    # plt.savefig("image")
    # plt.imshow(mask_manual_np[:,:,0], cmap='gray')
    # plt.savefig("label")

    image_np = image_np.astype(np.int16)
    mask_manual_np =mask_manual_np.astype(np.int16)

#     # manual mask
#     # non-std
# #     print(image_np.shape)
# #     print(mask_manual_np.shape)
    t = 0
    for i,i2 in tqdm(zip(image_np,mask_manual_np),total = len(image_np)):
        t = t + 1

        try:
            feature_values, feature_columns = feature_extract(i, i2)
            
            feature_dict = {}
            feature_dict["pid"] = pf_name
            feature_dict["Slice_num"] = t

            for val,feat in zip(feature_columns, feature_values):
                feature_dict[val] = feat

            pindex = pf_name + "_" + str(file_name)

            if First:
                First = False
                total_df = pd.DataFrame(feature_dict, index=[pindex])
            else:
                temp_df = pd.DataFrame(feature_dict, index=[pindex])
                total_df = total_df.append(temp_df, sort = True)

            total_df.to_csv('./radiomics_normalize_(8,128).csv')

        except Exception as e:
            # print(e)
            error_list.append(pf_name+"_"+file_name+"_Manual")
            # print(error_list)
            pass

    

    # Add Non-STD Results to Pandas DataFrame
    

# # print(len(error_list))
# # print(error_list)
# # print(cnt)