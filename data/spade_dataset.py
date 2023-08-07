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


class PainWomaskDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader):
        imgs = sorted(glob.glob(data_root))  # images data root

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
            self.imgs = [imgs[i] for i in range(len(imgs)) if i not in ids]

        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
            # A.ToTensor()
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.model = torch.load('submodels/atten_0706.pth', map_location='cpu').eval()
        # self.model = torch.load('submodels/atten_0706.pth', map_location='cpu').eval()

        self.kernal_size = 13
        self.conv = nn.Conv2d(1, 1, self.kernal_size, 1, self.kernal_size // 2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = 13
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

        cond_image = tiff.imread(path.replace('bp', 'ap'))
        cond_image = (cond_image - cond_image.min()) / (cond_image.max() - cond_image.min())
        cond_image = (cond_image - 0.5) / 0.5
        transformed = self.tfs(image=cond_image)
        cond_image = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)

        mask, cond_image, pred_mask = self.transform_mask(
            tiff.imread(path.replace('bp', 'apeff/apeff').replace('/ap/', '/apeff/apeff/')),
            tiff.imread(path.replace('bp', 'apmean').replace('/ap/', '/apmean/').replace('.', '_100.0.')),
            cond_image)
        mask = mask.unsqueeze(0)

        ret['mask'] = mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = cond_image
        ret['pred_mask'] = pred_mask
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
            threshold = 0.07 * self.kernal_size_2 * self.kernal_size_2
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

            if img is not None:
                cond_img = (1 - mask) * img + mask * torch.randn_like(img)

                img = torch.unsqueeze(img, 0)
                pred = self.model(img)
                pred = torch.argmax(pred, 1, True)

                # tiff.imwrite('pred.tif', to_8bit(pred))
                # tiff.imwrite('see.tif', to_8bit(masked_inside))
                # tiff.imwrite('seeee.tif', to_8bit(masked_outside))

        return torch.ones((img.shape[1], img.shape[2])), cond_img, pred

if __name__ == "__main__":
    train_dataset = PainWomaskDataset("/media/ziyi/Dataset/OAI_pain/full/ap/*", mask_config={"mask_mode": "hybrid"})

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True, shuffle=False)
    for i in train_dataloader:
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        assert 0