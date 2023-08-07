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


class PainValidationDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader):
        all = sorted(glob.glob(data_root))
        ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]

        imgs = [all[i] for i in ids]

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
            # A.ToTensor()
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

        # self.model = torch.load('submodels/80.pth', map_location='cpu').eval()
        self.kernal_size = 11
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

        transformed = self.tfs(image=img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        mask = torch.unsqueeze(
            self.transform_mask(tiff.imread(path.replace('ap', 'apeff/apeff')),
                                tiff.imread(path.replace('ap', 'apmean').replace('.', '_100.0.'))), 0)

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

    def transform_mask(self, mask, mask_2=None, img=None):
        threshold = 0

        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)

        if mask_2 is not None:
            threshold = 0.04 * self.kernal_size_2 * self.kernal_size_2 # 0.04
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask = mask_2 - mask
            # mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)

        contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contour)
        for num, i in enumerate(contour):
            print(cv2.contourArea(i))
        #     blank = np.zeros(mask.shape[:], dtype='uint8')
        #     cv2.drawContours(blank, i, -1, (255, 0, 0), 1)
        #     cv2.imwrite(f"img_{num}.png", blank)
        #     cv2.imwrite("mask.png", mask * 255)
        blank = np.zeros(mask.shape[:], dtype='uint8')
        cv2.drawContours(blank, contour, -1, (255, 0, 0), 1)
        cv2.imwrite(f"img.png", blank)



        assert 0
        # cv2.waitKey(0)
        return mask

if __name__ == "__main__":
    train_dataset = PainValidationDataset("/media/ziyi/Dataset/OAI_pain/full/ap/*", mask_config={"mask_mode": "hybrid"})

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True, shuffle=False)
    for i in train_dataloader:
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        assert 0