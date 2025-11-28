import numpy as np
import SimpleITK as sitk
import os
import open3d as o3d

CSV_root = r"H:"
PC_root = r"H:"
CT_Body_root = r"H:"
save_path = r"H:"

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
    else:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
    return Resampleimage

PC_path = os.path.join(PC_root, "PC.txt")
array_1 = np.loadtxt(PC_path)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(array_1)
# pcd.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([pcd])

# print(array_1.shape)
CT_Body_path = os.path.join(CT_Body_root, "mask_Body_50.nii")
CT_Body_img = sitk.ReadImage(CT_Body_path)
CT_Body_array=sitk.GetArrayFromImage(CT_Body_img)
size = np.array(CT_Body_img.GetSize())
spacing = np.array(CT_Body_img.GetSpacing())
origin = CT_Body_img.GetOrigin()


TPS_couch_shift = np.loadtxt(os.path.join(CSV_root, "couch_shift.txt"))
array_1 = array_1 - (TPS_couch_shift[[0, 2, 1]]) * np.array([-1, -1, 1])
array_1 = (array_1 / (np.array([spacing[0], spacing[2], -spacing[1]])))
array_1[:, [0, 1, 2]] = (array_1[:, [1, 2, 0]])
array_1 = np.around(array_1)

ref_index = np.loadtxt(os.path.join(CSV_root, "ref_index.txt"))
array_1 = array_1 + ref_index
# array_1 = array_1 + np.array([413, 269, 256])
delete_index_1 = np.array(np.where(array_1 < 0))
array_1 = np.delete(array_1, delete_index_1[0, :], 0)
delete_index_2 = np.array(np.where(array_1 > (size[1] - 1)))
array_1 = np.delete(array_1, delete_index_2[0, :], 0)
delete_index_3 = np.array(np.where(array_1[:, 0] > size[2] - 1))
array_1 = np.delete(array_1, delete_index_3[0, :], 0).astype(np.int32)


empty = np.zeros((size[2], size[1], size[0]))
# empty[50,100,150]=1
# array_2=np.array([[10,20,30],[40,50,50],[20,40,40]])
for i in range(array_1.shape[0]):
    empty[array_1[i][0], array_1[i][1], array_1[i][2]] = 1
CT_write = sitk.GetImageFromArray(empty)
# print(CT_write.GetSize())
CT_write.SetSpacing(spacing)
CT_write.SetOrigin(origin)
sitk.WriteImage(CT_write, os.path.join(save_path, "mask_OSI_0.nii"))

for i_expand in range(2):
    CT_write = sitk.Expand(CT_write, [3, 3, 3], sitk.sitkLinear)
# CT_write=sitk.Threshold(CT_write,-1,0,1)
    CT_write = sitk.BinaryThreshold(CT_write, lowerThreshold=0.00001, upperThreshold=1, insideValue=1,
                                            outsideValue=0)
    CT_write=sitk.Cast(CT_write,sitk.sitkFloat32)
    CT_write = Resampling(CT_write, spacing, lable=True)
CT_write.SetSpacing(spacing)
CT_write.SetOrigin(origin)
sitk.WriteImage(CT_write, os.path.join(save_path, "mask_OSI_1.nii"))

array_2=sitk.GetArrayFromImage(CT_write)
for z in range(array_2.shape[0]):
    for y in range(array_2.shape[1]):
        p = np.array(np.where(array_2[z, y, :] == 1))
        if not np.size(p) == 0:
            x_min = np.min(p)
            x_max = np.max(p)
            if x_max - x_min - 1>0:
                x = np.linspace(x_min + 1, x_max - 1, x_max - x_min - 1).astype(np.int32)
                # for i in range(y.shape[0]):
                #     print("z,y,x:",z,y[i],x)
                #     empty[z, y[i], x] = 1
                array_2[z, y, x] = 1

# array_2[CT_Body_array==0]=0

CT_write=sitk.GetImageFromArray(array_2)
CT_write.SetSpacing(spacing)
CT_write.SetOrigin(origin)
CT_write = sitk.Cast(CT_write, sitk.sitkInt16)
sitk.WriteImage(CT_write, os.path.join(save_path, "mask_OSI_2.nii"))

CT_write=sitk.BinaryMorphologicalClosing(CT_write,8)
CT_write=sitk.BinaryMorphologicalClosing(CT_write,8)
sitk.WriteImage(CT_write, os.path.join(save_path, "mask_OSI_3.nii"))

