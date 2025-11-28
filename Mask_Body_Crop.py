import SimpleITK as sitk
import os
import numpy as np
import open3d as o3d

def Resampling(img, NEW_spacing, lable=False):
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()
    new_spacing = NEW_spacing

    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    resampleSliceFilter = sitk.ResampleImageFilter()
    if lable == False:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkBSpline,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
        # Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
        #                                              img.GetOrigin(), new_spacing, img.GetDirection(), 0,
        #                                              img.GetPixelIDValue())
        # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
        # ResampleimageArray[ResampleimageArray < 0] = 0
    else:  # for label, should use sitk.sitkLinear to make sure the original and resampled label are the same!!!
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
    # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
    return Resampleimage

root=r""
new_spacing=[1,1,1]
img_body=sitk.ReadImage(os.path.join(root,"mask_Body.nii"))
img_body=Resampling(img_body,new_spacing,lable=True)
spacing=img_body.GetSpacing()
origin=img_body.GetOrigin()
array_body=sitk.GetArrayFromImage(img_body)
array_body[:,301:,:]=0
array_body[0,:,:]=0
img_body=sitk.GetImageFromArray(array_body)
img_body.SetSpacing(spacing)
img_body.SetOrigin(origin)
sitk.WriteImage(img_body,r"")