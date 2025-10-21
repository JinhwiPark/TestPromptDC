
import numpy as np
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