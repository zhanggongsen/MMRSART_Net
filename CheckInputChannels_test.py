import os
import SimpleITK as sitk
import numpy as np

test_path=r"H:"
img=sitk.ReadImage(os.path.join(test_path,"stacked_img_MTL0.nii"))
array=sitk.GetArrayFromImage(img)
print(array.shape)
for i in range(array.shape[0]):
    array_i= array[i]
    img_out=sitk.GetImageFromArray(array_i)
    sitk.WriteImage(img_out,os.path.join(test_path,"channel_{}.nii".format(str(i))))
