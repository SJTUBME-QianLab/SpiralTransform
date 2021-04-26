# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

import torchvision.transforms as transforms

import random

MODEL_DIR = '/result/'


class DataSet(Dataset):
    def __init__(self, data_dir, image_list_file, fold, transform=None, fold_num=None, filepath=None, mode='val'):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            fold: five fold index list of dataset
            transform: optional transform to be applied on a sample.
            fold_num: the number of fold
            filepath: to save image_name.txt  (default: args)
        """
        image_names = []
        image_names_original = []
        labels, image1, image2, image3 = [], [], [], []
        self.datalen = len(fold)
        fileline = open(image_list_file, "r").readlines()
        self.angle_dict_1 = {1: '_000000_', 2: '_000hor_', 3: '_000ver_', 4: '_l10_', 5: '_l10hor_', 6: '_l10ver_',
                             7: '_r10_', 8: '_r10hor_', 9: '_r10ver_', 10: '_l20_', 11: '_l20hor_', 12: '_l20ver_',
                             13: '_r20_', 14: '_r20hor_', 15: '_r20ver_', 16: '_l30_', 17: '_l30hor_', 18: '_l30ver_',
                             19: '_r30_', 20: '_r30hor_', 21: '_r30ver_', 22: '_l40_', 23: '_l40hor_', 24: '_l40ver_',
                             25: '_r40_', 26: '_r40hor_', 27: '_r40ver_'} # , 28: '_l50_', 29: '_l50hor_'}
        self.angle_dict_0 = {1: '_000000_', 2: '_000hor_', 3: '_000ver_', 4: '_l10_', 5: '_l10hor_', 6: '_l10ver_',
                             7: '_r10_', 8: '_r10hor_', 9: '_r10ver_', 10: '_l20_', 11: '_l20hor_', 12: '_l20ver_',
                             13: '_r20_', 14: '_r20hor_', 15: '_r20ver_', 16: '_l30_', 17: '_l30hor_', 18: '_l30ver_',
                             19: '_r30_', 20: '_r30hor_', 21: '_r30ver_', 22: '_l40_', 23: '_l40hor_', 24: '_l40ver_',
                             25: '_r40_', 26: '_r40hor_',  27: '_r40ver_'}
        for i in range(self.datalen):
            line = fileline[fold[i]]
            items = line.split()
            label = items[3]
            label = [int(i) for i in label]
            if label[0] == 1:
                for j in range(1, len(self.angle_dict_1)+1):
                    image_name = [os.path.join(data_dir, items[0])+self.angle_dict_1[j],
                                  os.path.join(data_dir, items[1])+self.angle_dict_1[j],
                                  os.path.join(data_dir, items[2])+self.angle_dict_1[j]]  # 没有文件格式的后缀
                    image_names.append(image_name)
                    labels.append(label)
            else:
                for j in range(1,len(self.angle_dict_0)+1):
                    image_name = [os.path.join(data_dir, items[0]) + self.angle_dict_0[j],
                                  os.path.join(data_dir, items[1]) + self.angle_dict_0[j],
                                  os.path.join(data_dir, items[2]) + self.angle_dict_0[j]]  # 没有文件格式的后缀
                    image_names.append(image_name)
                    labels.append(label)
            image_names_original.append(items[0])

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

        normalize_ADC = transforms.Normalize([0.35992014, 0.35992014, 0.35992014], [0.15849279, 0.15849279, 0.15849279])
        normalize_DWI = transforms.Normalize([0.23943327, 0.23943327, 0.23943327], [0.19963537, 0.19963537, 0.19963537])
        normalize_T2 = transforms.Normalize([0.25435543, 0.25435543, 0.25435543], [0.17792866, 0.17792866, 0.17792866])

        self.transform_ADC = transforms.Compose(
            [transforms.ToTensor(), normalize_ADC])  # transforms.RandomChoice(transforms_list),
        self.transform_DWI = transforms.Compose(
            [transforms.ToTensor(), normalize_DWI])
        self.transform_T2 = transforms.Compose(
            [transforms.ToTensor(), normalize_T2])

        if fold_num is not None:
            filename = MODEL_DIR + "{}/fold_{}_image_names_{}.txt".format(filepath.name, str(fold_num), mode)
            file = open(filename, 'w')  # 'a' 新建  'w+' 追加
            for w in range(self.datalen):
                # print(image_names[w][0])
                out_write = image_names_original[w][0:9] + '\n'
                file.write(out_write)
            file.close()

    def __getitem__(self, index):  # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """

        # image = Image.open(image_name).convert('RGB')
        # print(index)
        label = self.labels[index]  # 存储标签
        image1 = Image.open(self.image_names[index][0] + '.png').convert('RGB')
        image2 = Image.open(self.image_names[index][1] + '.png').convert('RGB')
        image3 = Image.open(self.image_names[index][2] + '.png').convert('RGB')
        if self.transform is not None:
            image1 = self.transform_ADC(image1)
            image2 = self.transform_DWI(image2)
            image3 = self.transform_T2(image3)
        # print(index)
        return image1, image2, image3, torch.tensor(label), index % len(self.image_names)

    def __len__(self):
        return len(self.image_names)


class DataSet_Mini(Dataset):  # 每一个batch里面的类别均衡
    def __init__(self, data_dir, image_list_file, fold, transform=None, fold_num=None, filepath=None, mode='val'):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            fold: five fold index list of dataset
            transform: optional transform to be applied on a sample.
            fold_num: the number of fold
            filepath: to save image_name.txt  (default: args)
        """
        self.image_names_0, self.image_names_1 = [], []
        image_names_original = []
        self.labels_0, self.labels_1 = [], []
        self.datalen = len(fold)
        fileline = open(image_list_file, "r").readlines()

        self.angle_dict_1 = {1: '_000000_', 2: '_000hor_', 3: '_000ver_', 4: '_l10_', 5: '_l10hor_', 6: '_l10ver_',
                             7: '_r10_', 8: '_r10hor_', 9: '_r10ver_', 10: '_l20_', 11: '_l20hor_', 12: '_l20ver_',
                             13: '_r20_', 14: '_r20hor_', 15: '_r20ver_', 16: '_l30_', 17: '_l30hor_', 18: '_l30ver_',
                             19: '_r30_', 20: '_r30hor_', 21: '_r30ver_', 22: '_l40_', 23: '_l40hor_', 24: '_l40ver_',
                             25: '_r40_', 26: '_r40hor_', 27: '_r40ver_'} # , 28: '_l50_', 29: '_l50hor_'}
        self.angle_dict_0 = {1: '_000000_', 2: '_000hor_', 3: '_000ver_', 4: '_l10_', 5: '_l10hor_', 6: '_l10ver_',
                             7: '_r10_', 8: '_r10hor_', 9: '_r10ver_', 10: '_l20_', 11: '_l20hor_', 12: '_l20ver_',
                             13: '_r20_', 14: '_r20hor_', 15: '_r20ver_', 16: '_l30_', 17: '_l30hor_', 18: '_l30ver_',
                             19: '_r30_', 20: '_r30hor_', 21: '_r30ver_', 22: '_l40_', 23: '_l40hor_', 24: '_l40ver_',
                             25: '_r40_', 26: '_r40hor_', 27: '_r40ver_'}
        for i in range(self.datalen):
            line = fileline[fold[i]]
            items = line.split()
            label = items[3]
            label = [int(i) for i in label]
            if label[0] == 1:
                for j in range(1, len(self.angle_dict_1)+1):
                    image_name = [os.path.join(data_dir, items[0])+self.angle_dict_1[j],
                                  os.path.join(data_dir, items[1])+self.angle_dict_1[j],
                                  os.path.join(data_dir, items[2])+self.angle_dict_1[j]]  # 没有文件格式的后缀
                    self.image_names_1.append(image_name)
                    self.labels_1.append(label)
            else:
                for j in range(1,len(self.angle_dict_0)+1):
                    image_name = [os.path.join(data_dir, items[0]) + self.angle_dict_0[j],
                                  os.path.join(data_dir, items[1]) + self.angle_dict_0[j],
                                  os.path.join(data_dir, items[2]) + self.angle_dict_0[j]]  # 没有文件格式的后缀
                    self.image_names_0.append(image_name)
                    self.labels_0.append(label)
            image_names_original.append(items[0])

        self.label0_index_box = np.arange(len(self.labels_0))  # 生成一串序号
        self.label1_index_box = np.arange(len(self.labels_1))
        self.transform = transform

        normalize_ADC = transforms.Normalize([0.35992014, 0.35992014, 0.35992014], [0.15849279, 0.15849279, 0.15849279])
        normalize_DWI = transforms.Normalize([0.23943327, 0.23943327, 0.23943327], [0.19963537, 0.19963537, 0.19963537])
        normalize_T2 = transforms.Normalize([0.25435543, 0.25435543, 0.25435543], [0.17792866, 0.17792866, 0.17792866])

        self.transform_ADC = transforms.Compose(
            [transforms.ToTensor(), normalize_ADC])  # transforms.RandomChoice(transforms_list),
        self.transform_DWI = transforms.Compose(
            [transforms.ToTensor(), normalize_DWI])
        self.transform_T2 = transforms.Compose(
            [transforms.ToTensor(), normalize_T2])

        if fold_num is not None:
            filename = MODEL_DIR + "{}/fold_{}_image_names_{}.txt".format(filepath.name, str(fold_num), mode)
            file = open(filename, 'w')  # 'a' 新建  'w+' 追加
            for w in range(self.datalen):
                # print(image_names[w][0])
                out_write = image_names_original[w][0:9] + '\n'
                file.write(out_write)
            file.close()

    def __getitem__(self, index):  # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """

        # 按照序号index返回数据和标签

        if index % 2 == 0:  # 无放回抽取一个0类
            length = len(self.label0_index_box)  # 获取一下盒子中剩余元素的个数
            if length == 0:  # 如果盒子为空，则重新填满
                self.label0_index_box = np.arange(len(self.labels_0))
                length = len(self.label0_index_box)
            random_index = random.randint(0, length - 1)  # 从盒子中剩余的样本中随机抽取一个
            image1 = Image.open(self.image_names_0[self.label0_index_box[random_index]][0] + '.png').convert('RGB')
            image2 = Image.open(self.image_names_0[self.label0_index_box[random_index]][1] + '.png').convert('RGB')
            image3 = Image.open(self.image_names_0[self.label0_index_box[random_index]][2] + '.png').convert('RGB')
            label = self.labels_0[self.label0_index_box[random_index]]
            self.label0_index_box = np.delete(self.label0_index_box, random_index)  # 将抽出的样本序号从盒子中删除
        else:  # index % 2 == 1:  # 无放回抽取一个1类
            length = len(self.label1_index_box)  # 获取一下盒子中剩余元素的个数
            if length == 0:  # 如果盒子为空，则重新填满
                self.label1_index_box = np.arange(len(self.labels_1))
                length = len(self.label1_index_box)
            random_index = random.randint(0, length - 1)  # 从盒子中剩余的样本中随机抽取一个
            image1 = Image.open(self.image_names_1[self.label1_index_box[random_index]][0] + '.png').convert('RGB')
            image2 = Image.open(self.image_names_1[self.label1_index_box[random_index]][1] + '.png').convert('RGB')
            image3 = Image.open(self.image_names_1[self.label1_index_box[random_index]][2] + '.png').convert('RGB')
            label = self.labels_1[self.label1_index_box[random_index]]
            self.label1_index_box = np.delete(self.label1_index_box, random_index)  # 将抽出的样本序号从盒子中删除

        if self.transform is not None:
            image1 = self.transform_ADC(image1)
            image2 = self.transform_DWI(image2)
            image3 = self.transform_T2(image3)
        # print(index)
        return image1, image2, image3, torch.tensor(label), index

    def __len__(self):
        return len(self.image_names_0) + len(self.image_names_1)
