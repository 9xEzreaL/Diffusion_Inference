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

# from model.unet import Unet

# from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def to_8bit(img):
    img = np.array(img)
    img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


class PainMonteCarloVer(data.Dataset):
    def __init__(self, data_root, mask_root=[], image_size=[384, 384], ids='all',
                 threshold=0.2, masking_mode='CNN', kernal_size=1):
        all = sorted(glob.glob(data_root))
        if ids == "all":
            self.imgs = all
        else:
            # ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
            self.imgs = [all[i] for i in ids]

        self.mask_root = mask_root
        assert len(self.mask_root)==1 or len(self.mask_root)==2 , f"Current mask should be one or two, but got {len(self.mask_root)} assigned"
        self.threshold = threshold
        self.masking_mode = masking_mode
        self.image_size = image_size

        self.kernal_size = kernal_size
        self.conv = nn.Conv2d(1, 1, self.kernal_size, 1, self.kernal_size // 2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = kernal_size
        self.conv_2 = nn.Conv2d(1, 1, self.kernal_size_2, 1, self.kernal_size_2 // 2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        id = path.rsplit("/")[-1].rsplit("\\")[-1]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        img = torch.unsqueeze(torch.Tensor(img), 0)
        if self.masking_mode == 'blur':
            mask = torch.unsqueeze(
                self.blur_mask(self.mask_root, id), 0)
        elif self.masking_mode == 'CNN':
            mask = torch.unsqueeze(
                self.transform_mask(self.mask_root, id), 0)

        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = id
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask_root, id):
        threshold = 0
        mask = tiff.imread(os.path.join(mask_root[0], id))
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if len(self.mask_root) == 2:
            threshold = self.threshold * self.kernal_size_2 * self.kernal_size_2 # 0.04
            mask_2 = tiff.imread(os.path.join(mask_root[1], id))
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            # mask = mask_2 - mask
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)
        return mask

    def blur_mask(self, mask_root, id):
        threshold = 0

        mask = tiff.imread(os.path.join(mask_root[0], id))
        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if len(self.mask_root) == 2:
            threshold = self.threshold
            mask_2 = tiff.imread(os.path.join(mask_root[1], id))
            mask_2 = cv2.GaussianBlur(mask_2, (self.kernal_size, self.kernal_size), 0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask = np.array(mask) + mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)
        return mask

    def get_mask(self, mask, mask_2=None, id=None):
        threshold = 0
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.2 * self.kernal_size_2 * self.kernal_size_2  # 0.07
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask = mask_2 - mask
            mask = np.array(mask > 0).astype(np.uint8) * 255

            # mask = torch.Tensor(mask)

        return mask


class PainAPMonteCarloVer(data.Dataset):
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
            self.transform_mask(tiff.imread(path.replace('ori', 'eff')),
                                tiff.imread(path.replace('ori', 'mean')),
                                id = index), 0)

        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask
        cond_image = torch.cat([cond_image, img], 0)  # (4, 384, 384)
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask, mask_2=None, img=None, id=None):
        threshold = 0

        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if img is not None:
            img = torch.unsqueeze(img, 0)
            pred = self.model(img)
            pred = torch.argmax(pred, 1, True)
            one_pred = (pred > 0).type(torch.uint8)

        if mask_2 is not None:
            threshold = 0.08 * self.kernal_size_2 * self.kernal_size_2 # 0.04
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            # mask = mask_2 - mask
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

            if img is not None:
                tmp = mask.type(torch.float32) - one_pred.type(torch.float32)
                masked_outside = (tmp > 0).type(torch.uint8) * torch.randn_like(img) + (1 - one_pred) * img
                masked_inside = one_pred * img + one_pred * (mask_2 > 0).type(torch.uint8) * torch.randn_like(img)

        if img is not None:
            return mask, masked_inside, masked_outside, pred
        else:
            return mask