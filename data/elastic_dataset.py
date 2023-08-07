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
        ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
        # ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
        # just for more sample to test
        # rand_ids = [random.randint(0, 29999) for y in range(40)]
        # ids += rand_ids

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = [imgs[i] for i in range(len(imgs)) if i not in ids]
        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
            # A.Rotate(limit=45, border_mode=0, value=0),
            A.ElasticTransform(interpolation=1, border_mode=0, value=0, alpha_affine=100, always_apply=True),
            # A.ToTensor()
        ]) # , additional_targets={'image0': 'image'}
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        # cond_image = tiff.imread(path.replace('bp', 'ap'))
        # cond_image = (cond_image - cond_image.min()) / (cond_image.max() - cond_image.min())

        transformed = self.tfs(image=img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        # cond_image = torch.unsqueeze(torch.Tensor(transformed["image0"]), 0)
        img = (img - 0.5) / 0.5
        # cond_image = (cond_image - 0.5) / 0.5
        mask = self.transform_mask(img)
        mask = mask.unsqueeze(0)
        # tiff.imwrite('img.tif', to_8bit(img))
        # tiff.imwrite('cond_img.tif', to_8bit(cond_image))

        # print(cond_image.shape)
        ret['mask'] = mask
        ret['gt_image'] = img
        ret['cond_image'] = img
        ret['mask_image'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, img):
        return torch.ones((img.shape[0], img.shape[1]))


if __name__ == "__main__":
    train_dataset = PainWomaskDataset("/media/ziyi/Dataset/OAI_pain/full/bp/*", mask_config={"mask_mode": "hybrid"})

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True, shuffle=False)
    for i in train_dataloader:
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        assert 0