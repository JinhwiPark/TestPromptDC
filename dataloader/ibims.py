import os 
import numpy as np
import torch.utils.data
from scipy import io
from PIL import Image


class valset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        with open(os.path.join(args.dataset_path, 'ibims1_core_mat/imagelist.txt')) as f:
            image_names = f.readlines()
            self.image_names = [x.strip() for x in image_names] 
            
        
    def __getitem__(self, index): # "/workspace/data_all/ibims1_core_mat/"
        image_data = io.loadmat(os.path.join(os.path.join(self.args.dataset_path, 'ibims1_core_mat'), self.image_names[index])) #  + self.image_names[index])

        #k_path = "/workspace/data_all/ibims1_core_raw/calib/" + self.image_names[index] + '.txt'
        k_path = os.path.join(os.path.join(self.args.dataset_path, 'ibims1_core_raw/calib'), self.image_names[index] + '.txt')
        if not os.path.exists(k_path):
            print('No K file')
            raise ValueError
        k = np.loadtxt(k_path, delimiter=',')

        data = image_data['data']
        
        rgb = data['rgb'][0][0] / 255.0
        depth = data['depth'][0][0]
        mask_invalid = data['mask_invalid'][0][0]
        mask_transp = data['mask_transp'][0][0]
        mask_missing = depth.copy()
        mask_missing[mask_missing!=0] = 1
        mask_valid = mask_invalid * mask_transp * mask_missing
        
        edges = data['edges'][0][0]
        
        rgb_np = np.array(rgb).astype(np.float32).transpose(2, 0, 1)
        depth_np = np.array(depth).astype(np.float32)
        
        depth_np = depth_np * mask_valid
        
        rgb_torch = torch.from_numpy(rgb_np).float()
        depth_torch = torch.from_numpy(depth_np).float().unsqueeze(0)
        sparse_depth = self.get_sparse_depth(depth_torch, 1000)
        
        k = torch.from_numpy(k).float()

        output = {'rgb': rgb_torch, 'dep': sparse_depth,
                  'gt': depth_torch, 'K': k}
        
        return output

    def __len__(self):
        return len(self.image_names)


    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.reshape(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))
    
        dep_sp = dep * mask.type_as(dep)

        return dep_sp

