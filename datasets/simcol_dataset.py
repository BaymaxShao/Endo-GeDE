from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
from scipy.spatial.transform import Rotation as R
import json
from torchvision import transforms
from PIL import Image


from .mono_dataset_2 import MonoDataset2


class SimColDataset(MonoDataset2):
    def __init__(self, *args, **kwargs):
        super(SimColDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.479, 0, 0.479, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SimColRAWDataset(SimColDataset):
    def __init__(self, *args, **kwargs):
        super(SimColRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        if frame_index - 1 == 0:
            frame_index += 1
        f_str = "FrameBuffer_{:04d}{}".format(frame_index-1, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)

        return image_path

    def get_depth(self, folder, frame_index, do_flip):
        f_str = "Depth_{:04d}{}".format(frame_index-1, self.img_ext)

        depth_path = os.path.join(self.data_path, folder, f_str)

        depth_gt = cv2.imread(depth_path)[:,:,0]/255/256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index):
        if frame_index - 1 == -1:
            frame_index += 1
        loc_reader = open(self.data_path + '/SavedPosition_{}.txt'.format(folder), 'r')
        rot_reader = open(self.data_path + '/SavedRotationQuaternion_{}.txt'.format(folder), 'r')
        location = list(map(float, loc_reader[frame_index - 1].split()))
        rotation = list(map(float, rot_reader[frame_index - 1].split()))
        location = np.array(location)
        rotation = np.array(rotation)
        r = R.from_quat(rotation).as_matrix()
        T = np.concatenate((r, location.reshape((3, 1))), 1)
        pose = np.concatenate((T, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)

        return pose


