# -*-coding=utf-8-*-

# radiomics
from featureExtractor import feature_extract
from glob import glob
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm 



# z-score Normalization

filepath = os.listdir("./preprocessed/")

def z_score_normalize(train_dataset):
    mean = np.mean(train_dataset)
    std = np.std(train_dataset)
    #print ("train data mean, std", mean, std)
    return mean, std

filepath = "./preprocessed/*"
save_dir= "./normalization/"

all_images = []
mean = []
std = []
num = 0
for pat in tqdm(glob(filepath), total = len(glob(filepath))): 
    pat_name = pat.split("/")[-1]
    print("Patient: ", pat_name)
    for file in glob(pat + "/*"):
        if 'label' not in file:
            file_name = file.split('/')[-1].split('.nii')[0]
            image_nii = sitk.ReadImage(file)
            image_np = sitk.GetArrayFromImage(image_nii)
            image_np = image_np.copy()
            mean1, std1 = z_score_normalize(image_np)
            mean.append(mean1)
            std.append(std1)
            num+=1
        else :
            pass

mean = np.sum(mean) / num # useless
std = np.sum(std) / num

print(num)
print(mean)
print(std)

for pat in tqdm(glob(filepath), total = len(glob(filepath))): 
    pat_name = pat.split("/")[-1]
    print("Patient: ", pat_name)
    for file in glob(pat + "/*"):
        if 'label' not in file:
            file_name = file.split('/')[-1].split('.nii')[0]
            image_nii = sitk.ReadImage(file)
            image_np = sitk.GetArrayFromImage(image_nii)
            image_np = image_np.copy()
            image_np_norm = (image_np - mean)/std
            print(file_name +':',np.max(image_np_norm),np.min(image_np_norm))
            image_np_norm = sitk.GetImageFromArray(image_np_norm)
            image_np_norm.CopyInformation(image_nii)
        else :
            file_name = file.split('/')[-1].split('.nii')[0]
            image_nii = sitk.ReadImage(file)
            image_np = sitk.GetArrayFromImage(image_nii)
            image_np = image_np.copy()
            image_np_norm = image_np
            print(file_name +':',np.max(image_np_norm),np.min(image_np_norm))
            image_np_norm = sitk.GetImageFromArray(image_np_norm)
            image_np_norm.CopyInformation(image_nii)

        save_dir = os.path.join('normalization2', str(pat_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if 'label' in file:
            lab_file_path = save_dir +  "/" +  file_name + ".nii"
            sitk.WriteImage(image_np_norm,lab_file_path)
            # np.save(lab_file_path, image_np_norm)
        else :
            img_file_path = save_dir +  "/" + file_name + ".nii"
            sitk.WriteImage(image_np_norm,img_file_path)
            # np.save(img_file_path, image_np_norm)

