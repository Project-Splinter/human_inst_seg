import os
import glob
import random 

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def aug_matrix(w1, h1, w2, h2):
    dx = (w2 - w1) / 2.0
    dy = (h2 - h1) / 2.0
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])

    scale = np.min([float(w2)/w1, float(h2)/h1]) # min | max

    M = get_affine_matrix(
        center = (w2 / 2.0, h2 / 2.0), 
        translate = (0, 0), 
        scale = scale)
    M = np.array(M + [0., 0., 1.]).reshape(3, 3)
    M = M.dot(matrix_trans)
    return M 


def get_affine_matrix(center, translate, scale):
    cx, cy = center
    tx, ty = translate

    M = [1, 0, 0,
         0, 1, 0]
    M = [x * scale for x in M]

    # Apply translation and of center translation: RSS * C^-1
    M[2] += M[0] * (-cx) + M[1] * (-cy)
    M[5] += M[3] * (-cx) + M[4] * (-cy)

    # Apply center translation: T * C * RSS * C^-1
    M[2] += cx + tx
    M[5] += cy + ty
    return M

class Dataset(object):
    def __init__(self, 
                 input_size=512, 
                 image_dir="./data/images", 
                 label_dir="./data/labels",
                 train=True,
                 ):
        super().__init__()
        self.input_size = input_size
        self.train = train

        image_names = [f for f in os.listdir(image_dir) if f[-3:]=="jpg"]
        image_files = [os.path.join(image_dir, f) for f in image_names]
        label_files = [
            os.path.join(
                label_dir, 
                f.replace(".jpg", "-removebg-preview.png")
            ) for f in image_names
        ]

        self.image_files = []
        self.label_files = []
        for image_file, label_file in zip(image_files, label_files):
            if os.path.exists(image_file) and os.path.exists(label_file):
                self.image_files.append(image_file)
                self.label_files.append(label_file)

        self.image_files = self.image_files
        self.label_files = self.label_files
                    
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        ])

        if self.train:
            self.image_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
            ])
        else:
            self.image_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.mask_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,))
        ])

        print (f"Dataset: {self.__len__()}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label_file = self.label_files[index]

        image = Image.open(image_file).convert("RGB")
        width, height = image.size
        mask = Image.open(label_file).split()[-1]
        mask = mask.resize((width, height), Image.BILINEAR)

        if self.train:
            image = self.color_aug(image)

        M = aug_matrix(width, height, self.input_size, self.input_size)
            
        M_inv = np.linalg.inv(M)
        M_inv = M_inv[0:2].reshape(-1).tolist()

        image = image.transform(
            (self.input_size, self.input_size), Image.AFFINE, M_inv, 
            Image.BILINEAR, fillcolor=(128, 128, 128))
        mask = mask.transform(
            (self.input_size, self.input_size), Image.AFFINE, M_inv, 
            Image.BILINEAR, fillcolor=(0,))

        if self.train and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        input = self.image_to_tensor(image).float()
        label = self.mask_to_tensor(mask).long().squeeze(0)

        return input, label

if __name__ == "__main__":
    import torchvision

    dataset = Dataset(
        input_size=256, 
        image_dir="./JPEGImages/", 
        label_dir="./RemoveBG/",
    )

    images = []
    for i in range(16):
        image, mask = dataset[i]
        images.append(image)
    images = torch.stack(images)
    
    input_norm = images * 0.5 + 0.5 #[-1, 1] -> [0, 1]
    torchvision.utils.save_image(
        input_norm,
        f"./example.jpg", 
        normalize=True, range=(0, 1), nrow=4, padding=10, pad_value=0.5
    )
