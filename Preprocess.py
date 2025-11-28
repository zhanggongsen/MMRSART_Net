import SimpleITK as sitk
import numpy as np
import os


# Hu-2-miu
def convert_hu_to_linear_attenuation(hu_array_0,
                                     MU_WATER=20, MU_AIR=0.02, MU_MAX=None):
    hu_array = np.array(hu_array_0)
    if MU_MAX is None:
        MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    # convert values
    hu_array *= (MU_WATER - MU_AIR) / 1000
    hu_array += MU_WATER
    hu_array /= MU_MAX
    np.clip(hu_array, 0., 1., out=hu_array)
    return hu_array * MU_MAX

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


def Unified_Grayscale(image, min, max):
    image_array = sitk.GetArrayFromImage(image)
    array_tobeUnidied_max = np.maximum(image_array, min)
    array_tobeUnidied_min = np.minimum(array_tobeUnidied_max, max)
    image_Unified = sitk.GetImageFromArray(array_tobeUnidied_min)
    image_Unified.SetSpacing(image.GetSpacing())
    image_Unified.SetOrigin(image.GetOrigin())
    return image_Unified


def CropImage(image0, cropsize, pad_value):
    array0 = sitk.GetArrayFromImage(image0)
    # (filepath, filename) = os.path.split(Masks_file_toProcess)
    Shape0 = array0.shape
    if Shape0[0] >= cropsize[0]:
        array0 = array0
    else:
        array0 = np.pad(array0, (((cropsize[0] - Shape0[0]) // 2, (cropsize[0] - Shape0[0]) // 2 + 1), (0, 0), (0, 0)),
                        "constant", constant_values=pad_value)
    if Shape0[1] >= cropsize[1]:
        array0 = array0
    else:
        array0 = np.pad(array0, ((0, 0), ((cropsize[1] - Shape0[1]) // 2, (cropsize[1] - Shape0[1]) // 2 + 1), (0, 0)),
                        "constant", constant_values=pad_value)
    if Shape0[2] >= cropsize[2]:
        array0 = array0
    else:
        array0 = np.pad(array0, ((0, 0), (0, 0), ((cropsize[2] - Shape0[2]) // 2, (cropsize[2] - Shape0[2]) // 2 + 1)),
                        "constant", constant_values=pad_value)
    Shape1 = array0.shape
    array_cropped = array0[((Shape1[0] - cropsize[0]) // 2):(((Shape1[0] + cropsize[0]) // 2)),
                    ((Shape1[1] - cropsize[1]) // 2):(((Shape1[1] + cropsize[1]) // 2)),
                    ((Shape1[2] - cropsize[2]) // 2):(((Shape1[2] + cropsize[2]) // 2))]
    if array_cropped.dtype == "float32":
        array_float32 = array_cropped
    else:
        array_float32 = array_cropped.astype("float32")
    # print("root_path:{},shape1:{},cropped_shape{}:".format(root_path,Shape1,array_cropped.shape))
    image_out = sitk.GetImageFromArray(array_float32)
    image_out.SetSpacing(image0.GetSpacing())
    image_out.SetOrigin(image0.GetOrigin())
    return image_out


read_path = r"H:"
save_path = r"H:"


if not os.path.exists(save_path):
    os.mkdir(save_path)

file_list = []
dir_list = []

# new_spacing = [3, 3, 3]
# crop_size=[64, 128, 128]
# new_spacing = [1, 1, 3]
# crop_size = [64, 320, 448]
new_spacing = [2, 2, 3]
crop_size = [64, 160, 224]

for root, dirs, files in os.walk(read_path):
    for dir_i in dirs:
        dirs_path = os.path.join(root, dir_i)
        if len(dirs_path) == (len(read_path) + 9):
            dir_list.append(dirs_path)

processed_datanum = 0
for path_toProcess in dir_list:
    # print(path_toProcess)
    patient_num = path_toProcess.split('\\')[-1]

    save_path_patient = os.path.join(save_path, patient_num)
    if not os.path.exists(save_path_patient):
        os.mkdir(save_path_patient)

    for i in ["structs_0", "structs_50"]:
        path_read = os.path.join(path_toProcess, i)

        # mask_list = ["image.nii", "3DDRR_origin.nii", "recons_origin.nii", "mask_Body.nii"]
        if "Thoracic" in read_path:
            mask_list = ["image.nii", "3DDRR_origin.nii", "recons_origin_no_noise.nii", "recons_origin_noise.nii",
                         "mask_GTV.nii", "mask_Lung", "mask_Body.nii"]
        else:
            mask_list = ["image.nii", "3DDRR_origin.nii", "recons_origin_no_noise.nii", "recons_origin_noise.nii",
                         "mask_GTV.nii", "mask_Body.nii"]
        assert mask_list[-1] == "mask_Body.nii", "mask_list does not ends with mask_Body.nii"
        for j in mask_list:
            if j == "mask_Body.nii":
                path_body = os.path.join(path_read, j)
                if os.path.exists(path_body):
                    pass
                else:
                    path_body_new = os.path.join(path_read, 'mask_BODY.nii')
                    # path_read_new=path_read.replace('BODY','Body')
                    os.rename(path_body, path_body_new)
                    # path_read=path_read_new
            else:
                pass

            body_mask = sitk.ReadImage(os.path.join(path_read, "mask_Body.nii"))
            body_array_origin = sitk.GetArrayFromImage(body_mask)

            if j == "image.nii":
                img_mask = sitk.Cast(sitk.ReadImage(os.path.join(path_read, j)), sitk.sitkFloat32)
                CT_array_origin = sitk.GetArrayFromImage(img_mask)
                CT_array_origin = convert_hu_to_linear_attenuation(CT_array_origin,
                                                                   MU_WATER=20, MU_AIR=0.02, MU_MAX=None)
                CT_array_origin[body_array_origin != 255] = 0.02
                # print(np.max(CT_array_origin),np.min(CT_array_origin))
                CT_img_remask = sitk.GetImageFromArray(CT_array_origin)
                CT_img_remask.SetSpacing(img_mask.GetSpacing())
                CT_img_remask.SetOrigin(img_mask.GetOrigin())
                CT_img_resampled = Resampling(CT_img_remask, new_spacing, lable=False)
                img_tobe_cropped = Unified_Grayscale(CT_img_resampled, 0.02,
                                                     49.97)
                img_cropped = CropImage(img_tobe_cropped, crop_size, pad_value=0.02)

            elif j == "3DDRR_origin.nii":
                DRR_img = sitk.ReadImage(os.path.join(os.path.join(save_path_patient, "Cropped_" + i[8:]), j))
                DRR_arary_origin = sitk.GetArrayFromImage(DRR_img)
                # DRR_arary_origin[body_array_origin != 255] = 0
                DRR_img_remask = sitk.GetImageFromArray(DRR_arary_origin)
                DRR_img_remask.SetSpacing(DRR_img.GetSpacing())
                DRR_img_remask.SetOrigin(DRR_img.GetOrigin())
                DRR_img_resampled = Resampling(DRR_img_remask, new_spacing, lable=False)
                # DRR_tobe_cropped = Unified_Grayscale(DRR_img_resampled, -600, 0)
                img_cropped = CropImage(DRR_img_resampled, crop_size, pad_value=0)

            elif (j == "recons_origin_noise.nii") or (j == "recons_origin_no_noise.nii"):
                recons_img = sitk.ReadImage(os.path.join(os.path.join(save_path_patient, "Cropped_" + i[8:]), j))

                recons_arary_origin = sitk.GetArrayFromImage(recons_img)

                # recons_arary_origin[body_array_origin != 255] = np.min(recons_arary_origin)
                recons_img_remask = sitk.GetImageFromArray(recons_arary_origin)
                recons_img_remask.SetSpacing(recons_img.GetSpacing())
                recons_img_remask.SetOrigin(recons_img.GetOrigin())

                recons_img_resampled = Resampling(recons_img_remask, new_spacing, lable=False)
                # recons_tobe_cropped = Unified_Grayscale(recons_img_resampled, -1000, 1500)
                img_cropped = CropImage(recons_img_resampled, crop_size,
                                        pad_value=np.min(sitk.GetArrayFromImage(recons_img_resampled)))

            elif j == "mask_Lung":
                Lung_R_img = sitk.Cast(sitk.ReadImage(os.path.join(path_read, j + "_R.nii")), sitk.sitkFloat32)
                Lung_L_img = sitk.Cast(sitk.ReadImage(os.path.join(path_read, j + "_L.nii")), sitk.sitkFloat32)
                Lung_origin=sitk.ReadImage(os.path.join(path_read, j + "_R.nii")).GetOrigin()
                Lung_spacing = sitk.ReadImage(os.path.join(path_read, j + "_R.nii")).GetSpacing()
                Lung_R_array = sitk.GetArrayFromImage(Lung_R_img)
                Lung_L_array = sitk.GetArrayFromImage(Lung_L_img)
                Lung_R_array[Lung_L_array==255]=255
                Lung_All_img=sitk.GetImageFromArray(Lung_R_array)
                Lung_All_img.SetOrigin(Lung_origin)
                Lung_All_img.SetSpacing(Lung_spacing)
                img_tobe_cropped = Resampling(Lung_All_img, new_spacing, lable=True)
                img_cropped = CropImage(img_tobe_cropped, crop_size, pad_value=0)

            else:
                img_mask = sitk.Cast(sitk.ReadImage(os.path.join(path_read, j)), sitk.sitkFloat32)
                # array_img = sitk.GetArrayFromImage(img_mask)
                # if j == "mask_GTV.nii":
                #     array_img[array_img == 255] = 600
                # elif j == "mask_Body.nii":
                #     array_img[array_img == 255] = 400
                # img_remask = sitk.GetImageFromArray(array_img)
                # img_remask.SetSpacing(img_mask.GetSpacing())
                # img_remask.SetOrigin(img_mask.GetOrigin())
                img_tobe_cropped = Resampling(img_mask, new_spacing, lable=True)
                img_cropped = CropImage(img_tobe_cropped, crop_size, pad_value=0)

            file_save = os.path.join(save_path_patient, "Cropped_" + i[8:])
            if not os.path.exists(file_save):
                os.mkdir(file_save)
            # print("ID:{},phase:{},mask:{},size:{}".format(patient_num,i,j,img_cropped.GetSize()))
            if j == "3DDRR_origin.nii":
                sitk.WriteImage(img_cropped, os.path.join(file_save, "3DDRR.nii"))
            elif j == "recons_origin_no_noise.nii":
                sitk.WriteImage(img_cropped, os.path.join(file_save, "recons_no_noise.nii"))
            elif j == "recons_origin_noise.nii":
                sitk.WriteImage(img_cropped, os.path.join(file_save, "recons_noise.nii"))
            elif j == "mask_Lung":
                sitk.WriteImage(img_cropped, os.path.join(file_save, "mask_Lung_All.nii"))
            else:
                sitk.WriteImage(img_cropped, os.path.join(file_save, j))

    print("ID:{}".format(patient_num))
    processed_datanum += 1
    print("Processed data_num {}/{}".format(processed_datanum, len(dir_list)))
