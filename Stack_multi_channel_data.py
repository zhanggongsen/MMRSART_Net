import SimpleITK as sitk
import numpy as np
import os


def change_value(array_Tochange, value0):
    array_Tochange[array_Tochange == 0] = 0
    array_Tochange[array_Tochange == 255] = 2000
    np_change_value = np.minimum(array_Tochange, value0)
    return np_change_value

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


file_list = []

dir_list = []

# dict_list = {"mask_GTV.nii": 1, "mask_Body.nii": 0.5}
# lables = list(dict_list.keys())

Root = r"H:"
for root, dirs, files in os.walk(Root):
    for dir_i in dirs:
        dirs_path = os.path.join(root, dir_i)
        if len(dirs_path) == (len(Root) + 9):
            dir_list.append(dirs_path)

new_spacing=[2,2,3]
for file_toProcess in dir_list:
    print(file_toProcess[-8:])
    for phase in ["0", "50"]:
        stack_array = []
        if phase=="0":
            another_phase="50"
        else:
            another_phase="0"
        stack_array = []
        CT_0_path = os.path.join(os.path.join(file_toProcess, 'Cropped_{}'.format(phase)), 'image.nii')
        CT_0_img = sitk.ReadImage(CT_0_path)
        CT_0_img_resamp=Resampling(CT_0_img, new_spacing, lable=False)
        CT_0_array = sitk.GetArrayFromImage(CT_0_img_resamp)
        CT_0_array_1 = (CT_0_array - 0.02) / (49.97 - 0.02)
        stack_array.append(CT_0_array_1)

        GTV_0_path = os.path.join(os.path.join(file_toProcess, 'Cropped_{}'.format(phase)), 'mask_GTV.nii')
        GTV_0_img = sitk.ReadImage(GTV_0_path)
        GTV_0_img_resamp=Resampling(GTV_0_img, new_spacing, lable=True)
        GTV_0_array = sitk.GetArrayFromImage(GTV_0_img_resamp)
        GTV_0_array_1 = ((GTV_0_array - np.min(GTV_0_array)) / (np.max(GTV_0_array) - np.min(GTV_0_array)))
        stack_array.append(GTV_0_array_1)

        CT_50_path = os.path.join(os.path.join(file_toProcess, 'Cropped_{}'.format(another_phase)), 'image.nii')
        CT_50_img = sitk.ReadImage(CT_50_path)
        CT_50_img_resamp = Resampling(CT_50_img, new_spacing, lable=False)
        CT_50_array = sitk.GetArrayFromImage(CT_50_img_resamp)
        # CT_50_array_1 = ((CT_50_array - np.min(CT_50_array)) / (np.max(CT_50_array) - np.min(CT_50_array)))
        CT_50_array_1 = (CT_50_array - 0.02) / (49.97 - 0.02)# HU[-1000,1500] -> 1500 * (20-0.02) / 1000+20 = 49.97
        stack_array.append(CT_50_array_1)

        GTV_50_path = os.path.join(os.path.join(file_toProcess, 'Cropped_{}'.format(another_phase)), 'mask_GTV.nii')
        GTV_50_img = sitk.ReadImage(GTV_50_path)
        GTV_50_img_resamp = Resampling(GTV_50_img, new_spacing, lable=True)
        GTV_50_array = sitk.GetArrayFromImage(GTV_50_img_resamp)
        GTV_50_array_1 = ((GTV_50_array - np.min(GTV_50_array)) / (np.max(GTV_50_array) - np.min(GTV_50_array)))
        stack_array.append(GTV_50_array_1)

        # DRR3D_0_path = os.path.join(os.path.join(file_toProcess, 'Cropped_0'), '3DDRlR.nii')
        # DRR3D_0_img = sitk.ReadImage(DRR3D_0_path)
        # DRR3D_0_array = sitk.GetArrayFromImage(DRR3D_0_img)
        # DRR3D_0_array_1 = ((DRR3D_0_array - np.min(DRR3D_0_array)) / (np.max(DRR3D_0_array) - np.min(DRR3D_0_array)))
        # stack_array.append(DRR3D_0_array_1)

        Body_0_path = os.path.join(os.path.join(file_toProcess, 'Cropped_{}'.format(phase)), 'mask_Body.nii')
        Body_0_img = sitk.ReadImage(Body_0_path)
        Body_0_img_resamp = Resampling(Body_0_img, new_spacing, lable=True)
        # Body_0_img_Erode = sitk.BinaryErode(Body_0_img_resamp, (15, 15, 0))
        # Body_0_img_Ring = sitk.Subtract(Body_0_img_resamp, Body_0_img_Erode)
        Body_0_array = sitk.GetArrayFromImage(Body_0_img_resamp)
        Bodys_0_array_1 = ((Body_0_array - np.min(Body_0_array)) / (np.max(Body_0_array) - np.min(Body_0_array)))
        Bodys_0_array_1[:, 250:, :] = 0
        stack_array.append(Bodys_0_array_1)

        recons_0_path = os.path.join(os.path.join(file_toProcess, 'Cropped_{}'.format(phase)), 'recons_noise.nii')
        recons_0_img = sitk.ReadImage(recons_0_path)
        recons_0_img_resamp = Resampling(recons_0_img, new_spacing, lable=True)
        recons_0_array = sitk.GetArrayFromImage(recons_0_img_resamp)
        recons_0_array_1 = ((recons_0_array - np.min(recons_0_array)) / (np.max(recons_0_array) - np.min(recons_0_array)))
        stack_array.append(recons_0_array_1)

        stack_arrays = np.stack(stack_array, axis=0).astype('float32')
        print("stack_arrays.shape:{}".format(stack_arrays.shape))
        out_img = sitk.GetImageFromArray(stack_arrays)

        out_path = os.path.join(os.path.join(file_toProcess, 'Stacked_image_for_MTL'))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path_file = os.path.join(out_path, "stacked_img_MTL_{}_{}.nii".format(file_toProcess[-8:],phase))
        sitk.WriteImage(out_img, out_path_file)
