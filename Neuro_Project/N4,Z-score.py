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
        for file in glob(pat +"/img/*"):
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
# z-score Normalization

filepath = os.listdir("./preprocessed/")

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
        file_name = file.split('/')[-1].split('.nii')[0]
        image_nii = sitk.ReadImage(file)
        image_np = sitk.GetArrayFromImage(image_nii)
        image_np = image_np.copy()
        mean1, std1 = z_score_normalize(image_np)
        mean.append(mean1)
        std.append(std1)
        num+=1

        image_np_norm = (image_np - mean1)/std1
        print(file_name +':',np.max(image_np_norm),np.min(image_np_norm))
        image_np_norm = sitk.GetImageFromArray(image_np_norm)
        image_np_norm.CopyInformation(image_nii)

        save_dir = os.path.join('normalization', str(pat_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        else :
            img_file_path = save_dir +  "/" + file_name + ".nii"
            sitk.WriteImage(image_np_norm,img_file_path)
            # np.save(img_file_path, image_np_norm)

mean = np.sum(mean) / num # useless
std = np.sum(std) / num

print(num)
print(mean)
print(std)