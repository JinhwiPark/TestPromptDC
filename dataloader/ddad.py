import os
from glob import glob
import os.path as osp
import numpy as np
import json
import random
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)
        
def read_depth(file_name):

    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

class valset(BaseDataset):
    def __init__(self, args):
        super(valset, self).__init__(args)

        self.args = args

        self.sample_list = []
        self.extra_info = []
        self.calib_list = []

        datapath = self.args.dataset_path

        image_list = sorted(glob(osp.join(datapath, 'rgb/*.png')))
        gt_list = sorted(glob(osp.join(datapath, '*gt/*.png')))
        hints_list = sorted(glob(osp.join(datapath, 'hints/*.png')))
        calibtxt_list = sorted(glob(osp.join(datapath, 'intrinsics/*.txt')))

        for i in range(len(image_list)):
            self.sample_list += [[image_list[i], gt_list[i], hints_list[i]]]
            self.extra_info += [[image_list[i].split('/')[-1], False]] 
            self.calib_list += [[calibtxt_list[i]]]
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        K = np.loadtxt(self.calib_list[idx][0])
        rgb = Image.open(self.sample_list[idx][0])
        gt = read_depth(self.sample_list[idx][1])
        depth = read_depth(self.sample_list[idx][2])

        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        rgb = TF.to_tensor(rgb)

        depth = TF.to_tensor(np.array(depth))
        gt = TF.to_tensor(np.array(gt))
        
        output = {'rgb': rgb, 'dep': depth, 'gt': gt, 'K': torch.Tensor(K), 'dep_ip': depth, 'idx':idx}
        
        return output
    
    