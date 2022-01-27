
from pprint import pprint
from typing import Iterable, List
import albumentations as A
import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import nn as nn
from torchvision import transforms



def get_transformer(face_policy: str, patch_size: int, net_normalizer: transforms.Normalize, train: bool):
    # Transformers and traindb
    if face_policy == 'scale':
        # The loader crops the face isotropically then scales to a square of size patch_size_load
        loading_transformations = [
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
            A.Resize(height=patch_size,width=patch_size,always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    elif face_policy == 'tight':
        # The loader crops the face tightly without any scaling
        loading_transformations = [
            A.LongestMaxSize(max_size=patch_size, always_apply=True),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    else:
        raise ValueError('Unknown value for face_policy: {}'.format(face_policy))

    if train:
        aug_transformations = [
            A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
                ]),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
                A.ImageCompression(quality_lower=50, quality_upper=99),
            ], )
        ]
    else:
        aug_transformations = []

    # Common final transformations
    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        ToTensorV2(),
    ]
    transf = A.Compose(
        loading_transformations + downsample_train_transformations + aug_transformations + final_transformations)
    return transf



