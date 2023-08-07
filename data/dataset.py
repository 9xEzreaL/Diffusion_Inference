import torch.utils.data as data
import albumentations as A
from torchvision import transforms
from PIL import Image
import cv2
import os
import glob
import torch
import numpy as np
import tifffile as tiff
import random
import torch.nn as nn

from core.base_dataset import BaseDataset

# from model.unet import Unet

def pil_loader(path):
    return Image.open(path).convert('RGB')

def to_8bit(img):
    img = np.array(img)
    img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img

class PainValidationDataset(BaseDataset):# data.Dataset
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader, ids='all'):
        all = sorted(glob.glob(data_root))
        if ids == "all":
            self.imgs = all
        else:
            # ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
            self.imgs = [all[i] for i in ids]

        self.image_size = image_size

        self.kernal_size = 5
        self.conv = nn.Conv2d(1, 1, self.kernal_size, 1, self.kernal_size // 2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = 5
        self.conv_2 = nn.Conv2d(1, 1, self.kernal_size_2, 1, self.kernal_size_2 // 2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5
        img = torch.unsqueeze(torch.Tensor(img), 0)

        mask = torch.unsqueeze(
            self.transform_mask(tiff.imread(path.replace('ap', 'apeff/apeff')),
                                tiff.imread(path.replace('ap', 'apmean').replace('.', '_100.0.'))), 0)
        mask = mask.squeeze(0)
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask, mask_2=None):
        threshold = 0
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.04 * self.kernal_size_2 * self.kernal_size_2 # 0.04
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

        return mask

    def __get_mask(self, mask, mask_2=None, id=None):
        threshold = 0
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.07 * self.kernal_size_2 * self.kernal_size_2  # 0.07
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask = mask_2 - mask
            mask = np.array(mask > 0).astype(np.uint8) * 255

            # mask = torch.Tensor(mask)

        return mask


class PainWomaskValidationDataset(BaseDataset): # data.Dataset
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader, ids='all'):
        all = sorted(glob.glob(data_root))
        if ids == "all":
            self.imgs = all
        else:
            self.imgs = [all[i] for i in ids]

        self.image_size = image_size

        self.kernal_size = 5
        self.conv = nn.Conv2d(1, 1, self.kernal_size, 1, self.kernal_size // 2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = 5
        self.conv_2 = nn.Conv2d(1, 1, self.kernal_size_2, 1, self.kernal_size_2 // 2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5
        img = torch.unsqueeze(torch.Tensor(img), 0)

        cond_image = img

        fake_ones_mask, real_mask = self.transform_mask(
            tiff.imread(path.replace('bp', 'apeff/apeff').replace('/ap/', '/apeff/apeff/')),
            tiff.imread(path.replace('bp', 'apmean').replace('/ap/', '/apmean/').replace('.', '_100.0.')))

        cond_image = (1 - real_mask) * cond_image + real_mask * torch.randn_like(img)

        fake_ones_mask = fake_ones_mask.unsqueeze(0)
        real_mask = real_mask.unsqueeze(0)

        ret['mask'] = fake_ones_mask
        ret['real_mask'] = real_mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = cond_image
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask, mask_2=None):
        threshold = 0
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.04 * self.kernal_size_2 * self.kernal_size_2  # 0.07
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

        return torch.ones((mask.shape[0], mask.shape[1])), mask


class PainValidationDatasetHan(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader, ids='all'):
        all = sorted(glob.glob(data_root))
        if ids == "all":
            self.imgs = all
        else:
            # ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
            self.imgs = [all[i] for i in ids]

        self.image_size = image_size

        self.kernal_size = 5
        self.conv = nn.Conv2d(1, 1, self.kernal_size, 1, self.kernal_size // 2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = 5
        self.conv_2 = nn.Conv2d(1, 1, self.kernal_size_2, 1, self.kernal_size_2 // 2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}

        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5
        img = torch.unsqueeze(torch.Tensor(img), 0)

        cond_image = img

        #fake_ones_mask, real_mask = self.transform_mask(
        #    tiff.imread(path.replace('/a/', '/m/')))

        mask = torch.from_numpy(tiff.imread(path.replace('/a/', '/m/')))
        mask = (mask > 0.1) / 1

        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask, mask_2=None):
        threshold = 0
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.04 * self.kernal_size_2 * self.kernal_size_2 # 0.04
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

        return mask

    def __get_mask(self, mask, mask_2=None, id=None):
        threshold = 0
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.07 * self.kernal_size_2 * self.kernal_size_2  # 0.07
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask = mask_2 - mask
            mask = np.array(mask > 0).astype(np.uint8) * 255

            # mask = torch.Tensor(mask)

        return mask


if __name__ == "__main__":
    train_dataset = PainValidationDataset("/media/ziyi/Dataset/OAI_pain/full/ap/*", mask_config={"mask_mode": "hybrid"})

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True, shuffle=False)
    for i in train_dataloader:
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        pass
