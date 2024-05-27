# 小狗分类数据集处理
import pandas as pd
import cv2 as cv
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision

ROOT = os.path.dirname(os.path.abspath(__file__))
NUM_CATEGORY = 120
VAL_COUNT_PER_CATEGORY = 66

transform = {
    "train": torchvision.transforms.Compose([
        # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
        # 然后，缩放图像以创建224x224的新图像
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        # 随机更改亮度，对比度和饱和度
        torchvision.transforms.ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4),
        # 添加随机噪声
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])]),
    "test": torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
}


def get_csv(path):
    """
    获取csv文件
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    return df


def get_all_category(df):
    """
    获取类别名与位序的映射
    :param df:
    :return:
    """
    category = df['breed'].unique()
    breed2id = {}
    id2breed = {}

    for i in range(len(category)):
        breed2id[category[i]] = i
        id2breed[i] = category[i]

    return breed2id, id2breed


def get_all_image(df, breed2id):
    """
    获取所有图片地址与id的映射
    :param df:
    :param breed2id:
    :return:
    """
    record = df.values.tolist()
    random.shuffle(record)
    record = [[os.path.join(ROOT, 'train', i[0]), breed2id[i[1]]] for i in record]
    img2id = dict(record)

    return img2id


def split_train_test(img2id, val_rate=0.1):
    train_img2id = {}
    val_img2id = {}

    tag = [0] * NUM_CATEGORY
    count = val_rate * VAL_COUNT_PER_CATEGORY

    for i, v in img2id.items():
        if tag[v] < count:
            val_img2id[i] = v
            tag[v] += 1
        else:
            train_img2id[i] = v
    return train_img2id, val_img2id


class DogDataset(Dataset):
    def __init__(self, img2id, mode='train', transform=None):
        self.img2id = img2id
        self.mode = mode
        if self.mode == 'train':
            self.transform = transform['train']
        else:
            self.transform = transform['test']

    def __getitem__(self, index):
        img_path = list(self.img2id.items())[index][0] + ".jpg"
        label = list(self.img2id.items())[index][1]
        img = Image.open(img_path)
        img = self.transform(img)
        if self.mode == 'test':
            return img
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.img2id)


def get_dataset():
    pd = get_csv(os.path.join(ROOT, 'labels.csv'))
    breed2id, id2breed = get_all_category(pd)
    img2id = get_all_image(pd, breed2id)
    train_img2id, val_img2id = split_train_test(img2id)
    train_dataset = DogDataset(train_img2id, 'train', transform=transform)
    val_dataset = DogDataset(val_img2id, 'train', transform=transform)
    return train_dataset, val_dataset
