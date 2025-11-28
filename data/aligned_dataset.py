import os.path
import random
# from data.base_dataset import BaseDataset, get_params, get_transform
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import SimpleITK as sitk
import numpy as np
import torch


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # Data Augmentation
        self.degrees = self.opt.rotation_degrees
        self.ifRotate_n90 = self.opt.ifRotate_n90
        self.ifCrop = self.opt.ifCrop
        # 初始化dataB和dataA各自的灰度区间
        self.dataB_gray_low = self.opt.gray_low_dataB
        self.dataB_gray_high = self.opt.gray_high_dataB
        self.dataA_gray_low = self.opt.gray_low_dataA
        self.dataA_gray_high = self.opt.gray_high_dataA

    def __getitem__(self, index):
        # read a image given a random integer index
        AB_path = self.AB_paths[index]


        AB_nii = sitk.ReadImage(AB_path)
        # AB = Image.open(AB_path).convert('RGB')    #<class 'SimpleITK.SimpleITK.Image'>
        AB_np1 = sitk.GetArrayFromImage(AB_nii)  # <class 'numpy.ndarray'>

        # Random crop for data augmentation
        if self.ifCrop == 1:
            # origin_index_x = random.randint(10, self.opt.load_size - self.opt.crop_size-10)
            origin_index_x = random.randint(0, self.opt.load_size - self.opt.crop_size)
            end_index_x = origin_index_x + self.opt.crop_size
            # origin_index_y = random.randint(10, self.opt.load_size - self.opt.crop_size-10)
            origin_index_y = random.randint(0, self.opt.load_size - self.opt.crop_size)
            end_index_y = origin_index_y + self.opt.crop_size
            # origin_index_z = random.randint(10, self.opt.load_size - self.opt.crop_size-10)
            origin_index_z = random.randint(0, self.opt.load_size - self.opt.crop_size)

            end_index_z = origin_index_z + self.opt.crop_size
            AB_np1 = AB_np1[:, origin_index_x:end_index_x, origin_index_y:end_index_y, origin_index_z:end_index_z]
        else:
            pass
        # if self.ifCrop == 1:
        #
        #     AB_np1 = AB_np1[:, 0:96, 0:96, 0:96]
        # else:
        #     pass

        # if self.ifRotate_n90 == 1:
        #     k_n90 = random.randint(0, 3)
        #     if k_n90 == 0:
        #         pass
        #     else:
        #         AB_np1 = np.rot90(AB_np1.swapaxes(1, 3), k_n90, axes=(1, 2)).swapaxes(1, 3)
        # else:
        #     pass

        if self.ifRotate_n90 == 1:
            ifRotate = random.randint(0, 1)
            if ifRotate == 0:
                pass
            else:
                AB_np1 = np.rot90(AB_np1.swapaxes(1, 3), 2, axes=(1, 2)).swapaxes(1, 3)
        else:
            pass
        # AB_img = Image.fromarray(AB_np1[0].astype(np.float))  # <class 'PIL.Image.Image'>
        # print(AB_img.size)
        # w, h = AB_img.size
        # h2 = int(h / 2)
        # A = AB_img.crop((0, 0, w, h2))
        # B = AB_img.crop((0, h2, w, h))
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, params=transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, params=transform_params, grayscale=(self.output_nc == 1))
        # A = A_transform(A)  # <class 'PIL.Image.Image'>
        # B = B_transform(B)
        # if 'rotate' in self.opt.preprocess:
        #     degrees = self.degrees
        #     rotation_angle = random.uniform(-degrees, degrees)
        #     A = A.rotate(rotation_angle, resample=Image.BICUBIC, fillcolor=A.getpixel((0, 0)))
        #     B = B.rotate(rotation_angle, resample=Image.BICUBIC, fillcolor=B.getpixel((0, 0)))
        # A_np2 = np.asarray(A).astype(np.float)  # <class 'numpy.ndarray'>
        # B_np2 = np.asarray(B).astype(np.float)


        # FBP_t
        A_np2 = AB_np1[4:, :, :, :]
        # CT_GroudTruth
        B_np2 = AB_np1[0, :, :, :]
        # TumorMask_GroudTruth
        C_np2 = AB_np1[1, :, :, :]
        # Planning CT
        D_np2 = AB_np1[2, :, :, :]
        # Planning GTV;
        E_np2 = AB_np1[3, :, :, :]

        A_np3 = np.tile(A_np2.astype(np.float32), (1, 1, 1, 1))
        B_np3 = np.tile(B_np2.astype(np.float32), (1, 1, 1, 1,))
        C_np3 = np.tile(C_np2.astype(np.float32), (1, 1, 1, 1,))
        D_np3 = np.tile(D_np2.astype(np.float32), (1, 1, 1, 1,))
        E_np3 = np.tile(E_np2.astype(np.float32), (1, 1, 1, 1,))

        # B_np3 = ((B_np3 / np.max(B_np3))) * dataB_gray_high

        dataB_gray_high = self.dataB_gray_high
        dataB_gray_low = self.dataB_gray_low
        dataA_gray_high = self.dataA_gray_high
        dataA_gray_low = self.dataA_gray_low

        A_tensor = Normalize_and_Standard_toTensor(A_np3, dataA_gray_low, dataA_gray_high, normalize=False).to(
            torch.float)
        B_tensor = Normalize_and_Standard_toTensor(B_np3, dataB_gray_low, dataB_gray_high, normalize=False).to(
            torch.float)
        C_tensor = Normalize_and_Standard_toTensor(C_np3, dataB_gray_low, dataB_gray_high, normalize=False).to(
            torch.float)
        D_tensor = Normalize_and_Standard_toTensor(D_np3, dataB_gray_low, dataB_gray_high, normalize=False).to(
            torch.float)
        E_tensor = Normalize_and_Standard_toTensor(E_np3, dataB_gray_low, dataB_gray_high, normalize=False).to(
            torch.float)

        return {'A': A_tensor, 'B': B_tensor, 'C': C_tensor, 'D': D_tensor, 'E': E_tensor, 'A_paths': AB_path,
                'B_paths': AB_path, 'C_paths': AB_path, 'D_paths': AB_path, 'E_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)


def Normalize_and_Standard_toTensor(input_np, gray_low, gray_high, normalize=False):
    if isinstance(input_np, np.ndarray):
        if input_np.dtype == np.float32:
            # rescale your np.float into a float between [0, 1]. range depends on the modality
            output_np = (input_np - gray_low) * (1 - 0) / (gray_high - gray_low)
            float__tensor = torch.from_numpy(output_np)
            if normalize == True:
                # normalize [0,1] tensor to [-1,1].
                output_tensor = (float__tensor - 0.5) / 0.5
            else:
                output_tensor = float__tensor
    return output_tensor
