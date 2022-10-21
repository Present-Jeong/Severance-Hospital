# import os 
# from tqdm.notebook import tqdm 

# filepath = os.listdir("./Radiomics_dataset/")
# for pat in tqdm(filepath, total = len(filepath)): 
#     pat_name = pat.split("/")[-1]
#     # print(pat_name)
#     list = os.listdir('Radiomics_dataset/'+pat_name)
#     # print(len(list))
#     if len(list) is not 3:
#         print(pat_name)
#     # for file in list:
#         # print(len(file))
#         # print(file)
#         # tt = file.split('_')
#         # if 't1GD' in file:
#         #     os.rename('Radiomics_dataset/'+pat_name +'/'+file,'Radiomics_dataset/'+pat_name+'/'+pat_name+'_T1GD_registered-label.nii')
#         # else :
#         #     pass
# # # os.remove()

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import radiomics

# print(radiomics.__file__)
# file = 'preprocessed/1402591/1402591_T1GD_registered-label.nii'
# tt = sitk.ReadImage(file)
# image_np = sitk.GetArrayFromImage(tt)[120,:,:]
# plt.imshow(image_np, cmap = 'gray')
# plt.savefig("image")


# tt2 = np.load('normalization/label/7864018_T1GD_registered-label.npy')[120,:,:]
# plt.imshow(tt2,cmap='gray')
# plt.savefig("label")

from glob import glob
mask_path = './preprocessed/*'
# all_masks = sorted(glob(mask_path))
# all_index = sorted([int(i.split('/')[-1].split('_')[0]) for i in all_masks])
# # all_index = sorted([i.split('/')[1].split('_')[0] for i in all_masks])
print(len(glob(mask_path)))

# filepath = './normalization/1402591/1402591_T1GD_registered.nii'
# tt=sitk.ReadImage(filepath)
# tt= sitk.GetArrayFromImage(tt)
# print(tt.size)
# for pat in glob(filepath): 
#     pat_name = pat.split("/")[-1]
#     print(pat)