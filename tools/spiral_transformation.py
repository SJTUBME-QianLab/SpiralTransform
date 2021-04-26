# encoding: utf-8

"""
code for spiral-transformation

run this code before training the model

"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import nrrd
import math
import matplotlib.pyplot as plt
import copy
import cv2
import time
import sys

dx=int(sys.argv[1])
dy=int(sys.argv[2])
dz=int(sys.argv[3])
print(dx,dy,dz)
# dx, dy, dz = 0, 0, 0
R = 60
N = 20  # fai转了5圈
# modal = 'DWI'


class DataSet():
    def __init__(self, data_dir, image_list_file, data_dir_new, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names, image_mask_names = [], []
        # image1 = []
        with open(image_list_file) as f:
            for line in f:
                items = line.split()
                image_name = [os.path.join(data_dir, items[0] + '.nrrd')]
                image_mask_name = [os.path.join(data_dir, items[0] + '-s.nrrd')]
                # start = time.clock()
                # image1.append(nrrd.read(image_name[0])[0].transpose(1, 0, 2))
                # print("Time used:", (time.clock() - start))
                image_names.append(image_name)
                image_mask_names.append(image_mask_name)

                # print(i)

        self.image_names = image_names
        self.image_mask_names = image_mask_names
        # self.image1 = image1
        self.data_dir_new = data_dir_new
        self.transform = transform

    def data_new(self, index):  # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """

        # image = Image.open(image_name).convert('RGB签
        if index < len(self.image_names) * 3:  # 000
            image1 = nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)

            if index < len(self.image_names):
                trans = '000000'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 2:
                trans = '000hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:  # 垂直
                trans = '000ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
                # print(image1_trans.all()==image1.all())
                # plt.imshow(image1[:, :, 60])
                # plt.show()
                # plt.imshow(image1_trans[:, :, 60])
                # plt.show()
        elif index < len(self.image_names) * 6:  # l10
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='l10')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)

            image1_mask = image_trans(image1_mask, angle='l10')
            if index < len(self.image_names) * 4:
                trans = 'l10'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 5:
                trans = 'l10hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l10ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 9:  # r10
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='r10')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r10')
            if index < len(self.image_names) * 7:
                trans = 'r10'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 8:
                trans = 'r10hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r10ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 12:  # l20
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='l20')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l20')
            if index < len(self.image_names) * 10:
                trans = 'l20'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 11:
                trans = 'l20hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l20ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 15:  # r20
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='r20')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r20')
            if index < len(self.image_names) * 13:
                trans = 'r20'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 14:
                trans = 'r20hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r20ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 18:  # l30
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='l30')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l30')
            if index < len(self.image_names) * 16:
                trans = 'l30'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 17:
                trans = 'l30hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l30ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 21:  # l30
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='r30')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r30')
            if index < len(self.image_names) * 19:
                trans = 'r30'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 20:
                trans = 'r30hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r30ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 24:  # l40
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='l40')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l40')
            if index < len(self.image_names) * 22:
                trans = 'l40'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 23:
                trans = 'l40hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l40ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 27:  # r40
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2), angle='r40')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r40')
            if index < len(self.image_names) * 25:
                trans = 'r40'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 26:
                trans = 'r40hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r40ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 30:  # l50
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='l50')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l50')
            if index < len(self.image_names) * 28:
                trans = 'l50'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 29:
                trans = 'l50hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l50ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 33:  # r50
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='r50')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r50')
            if index < len(self.image_names) * 31:
                trans = 'r50'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 32:
                trans = 'r50hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r50ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 36:  # l60
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='l60')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l60')
            if index < len(self.image_names) * 34:
                trans = 'l60'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 35:
                trans = 'l60hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l60ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 39:  # r60
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='r60')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r60')
            if index < len(self.image_names) * 37:
                trans = 'r60'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 38:
                trans = 'r60hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r60ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 42:  # l70
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='l70')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l70')
            if index < len(self.image_names) * 40:
                trans = 'l70'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 41:
                trans = 'l70hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l70ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 45:  # r70
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='r70')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r70')
            if index < len(self.image_names) * 43:
                trans = 'r70'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 44:
                trans = 'r70hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r70ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        elif index < len(self.image_names) * 48:  # l80
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='l80')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='l80')
            if index < len(self.image_names) * 46:
                trans = 'l80'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 47:
                trans = 'l80hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l80ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        # elif index < len(self.image_names) * 51:  # r80
        else:
            image1 = image_trans(nrrd.read(self.image_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2),
                                 angle='r80')
            image1_mask = nrrd.read(self.image_mask_names[index % len(self.image_names)][0])[0].transpose(1, 0, 2)
            image1_mask = image_trans(image1_mask, angle='r80')
            if index < len(self.image_names) * 49:
                trans = 'r80'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 50:
                trans = 'r80hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r80ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')

        image1_slice = rotate_slice_evenly(image1_trans, self.image_names[index % len(self.image_names)][0],
                                           trans, image1_mask_trans, self.data_dir_new)

        print(index)
        return image1_slice

    def data_length(self):
        return len(self.image_names) * 27


def setpara(image):
    xx, yy, zz = image.shape
    mid_x = int(xx / 2)
    mid_y = int(yy / 2)
    mid_z = int(zz / 2)
    return mid_x, mid_y, mid_z


def image_trans(im1, angle):
    image1 = copy.deepcopy(im1)
    mid_x1, mid_y1, mid_z1 = setpara(image1)
    if angle == 'l10' or angle == 'l20' or angle == 'l30' or angle == 'l40' or \
            angle == 'l50' or angle == 'l60' or angle == 'l70' or angle == 'l80':  # 顺时针
        angle_num = int(angle[1:])
        rot_mat = cv2.getRotationMatrix2D((mid_y1, mid_x1), -angle_num, 1)
        for i1 in range(image1.shape[2]):
            image1[:, :, i1] = cv2.warpAffine(image1[:, :, i1], rot_mat,
                                              (image1[:, :, i1].shape[1], image1[:, :, i1].shape[0]))
    elif angle == 'r10' or angle == 'r20' or angle == 'r30' or angle == 'r40' or \
            angle == 'r50' or angle == 'r60' or angle == 'r70' or angle == 'r80':  # 逆时针
        angle_num = int(angle[1:])
        rot_mat = cv2.getRotationMatrix2D((mid_y1, mid_x1), angle_num, 1)
        for i1 in range(image1.shape[2]):
            image1[:, :, i1] = cv2.warpAffine(image1[:, :, i1], rot_mat,
                                              (image1[:, :, i1].shape[1], image1[:, :, i1].shape[0]))
    elif angle == 'flip_horizontally':  # 水平翻转
        for i1 in range(image1.shape[2]):
            image1[:, :, i1] = cv2.flip(image1[:, :, i1], 1)
    elif angle == 'flip_vertically':  # 垂直翻转
        for i1 in range(image1.shape[2]):
            image1[:, :, i1] = cv2.flip(image1[:, :, i1], 0)
    else:
        print('please choose right angle')

    return image1


def rotate_slice_evenly(nrrd_data, nrrd_filename, trans, image_mask, DATA_DIR_NEW):
    xx, yy, zz = image_mask.nonzero()
    # mid_dx = math.floor((max(xx)-min(xx))*1/30 * dx)
    # mid_dy = math.floor((max(yy)-min(yy))*1/30 * dy)
    # mid_dz = math.floor((max(zz)-min(zz))*1/30 * dz)
    # print(mid_dx, mid_dy, mid_dz)
    mid_dx = math.floor((max(xx)-min(xx))*1/40 * dx)
    mid_dy = math.floor((max(yy)-min(yy))*1/40 * dy)
    mid_dz = math.floor((max(zz)-min(zz))*1/40 * dz)
    mid_x = int(round((max(xx) + min(xx)) / 2)) + mid_dx
    mid_y = int(round((max(yy) + min(yy)) / 2)) + mid_dy
    mid_z = int(round((max(zz) + min(zz)) / 2)) + mid_dz
    # 以mid_x, mid_y, mid_z为坐标原点建立坐标系

    k = int(2 * N * N / np.pi)  # 4 * N * N / np.pi
    rotate_slice = np.zeros((2 * R, k))
    for r in range(-R, R):
        for n in range(0, k):
            # theta = np.pi/2 - (np.pi*n)/(2*k)
            # fai = N*2*np.pi*n/k
            if n == 0:
                x_coordinate = int(0)
                y_coordinate = int(0)
                z_coordinate = int(r)
            else:
                theta = n * np.pi * np.pi / (4 * N * N)
                fai = n * (np.pi / N) / (np.sin(theta))
                x_coordinate = r * np.sin(theta) * np.cos(fai)
                y_coordinate = r * np.sin(theta) * np.sin(fai)
                z_coordinate = r * np.cos(theta)
            # print(x_coordinate, y_coordinate, z_coordinate)
            if x_coordinate + mid_x >= nrrd_data.shape[0] - 1 or y_coordinate + mid_y >= nrrd_data.shape[1] - 1 or \
                    z_coordinate + mid_z >= nrrd_data.shape[2] - 1:
                continue
            else:
                # 插值求灰度
                xmim, ymin, zmin = math.floor(x_coordinate + mid_x), math.floor(y_coordinate + mid_y), \
                                   math.floor(z_coordinate + mid_z)
                xd = x_coordinate + mid_x - xmim
                yd = y_coordinate + mid_y - ymin
                zd = z_coordinate + mid_z - zmin
                c000, c100, c010, c001, c101, c011, c110, c111 = nrrd_data[xmim, ymin, zmin], \
                                                                 nrrd_data[xmim + 1, ymin, zmin], \
                                                                 nrrd_data[xmim, ymin + 1, zmin], \
                                                                 nrrd_data[xmim, ymin, zmin + 1], \
                                                                 nrrd_data[xmim + 1, ymin, zmin + 1], \
                                                                 nrrd_data[xmim, ymin + 1, zmin + 1], \
                                                                 nrrd_data[xmim + 1, ymin + 1, zmin], \
                                                                 nrrd_data[xmim + 1, ymin + 1, zmin + 1]
                rotate_slice[r + R, n] = c000 * (1 - xd) * (1 - yd) * (1 - zd) + \
                                         c100 * xd * (1 - yd) * (1 - zd) + c010 * (1 - xd) * yd * (1 - zd) + \
                                         c001 * (1 - xd) * (1 - yd) * zd + c101 * xd * (1 - yd) * zd + \
                                         c011 * (1 - xd) * yd * zd + c110 * xd * yd * (1 - zd) + c111 * xd * yd * zd

    rotate_slice_result = (rotate_slice-rotate_slice.min())/(rotate_slice.max()-rotate_slice.min())
    cv2.imwrite(DATA_DIR_NEW + nrrd_filename[51:-5] + '_' + trans + '_' + '.png', rotate_slice_result * 255)
    return rotate_slice


DATA_DIR = 'D:/data/data_Pancreatic_cancer/cancer/original_all/'
# DATA_DIR_NEW = 'D:/data/data_Pancreatic_cancer/cancer/data_classifier/data_rotate_R{}N{}/'.format(R,N)
DATA_DIR_NEW = 'D:/data/data_Pancreatic_cancer/cancer/data_classifier/data_rotate_{}{}{}/'.format(dx,dy,dz)
if not os.path.exists(DATA_DIR_NEW):
    os.makedirs(DATA_DIR_NEW)

# DATA_IMAGE_LIST = './label/original/TP53_3D_{}.txt'.format(modal)
DATA_IMAGE_LIST = 'D:/data/data_Pancreatic_cancer/cancer/label/TP53_3D_ADC_DWI_T2.txt'
dataset = DataSet(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, data_dir_new=DATA_DIR_NEW)
for idx in range(0, dataset.data_length()):
    im1 = dataset.data_new(idx)
