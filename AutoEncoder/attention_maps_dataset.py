from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os


class AttentionMapsDataset(Dataset):

    def __init__(self, data_dir, transform):
        self.data = []
        self.transform = transform

        for root, _, files in os.walk(data_dir): 
            for f in files:
                if f.endswith('.npy'):
                    self.data.append(os.path.join(root, f))

    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, index):
        path = self.data[index]

        attn_map = np.load(path).astype(np.float32)

        # Added new axis
        if attn_map.ndim == 2:
            attn_map = np.expand_dims(attn_map, 0)

        attn_map_tensor = torch.from_numpy(attn_map)

        img = self.transform(attn_map_tensor) if self.transform else attn_map_tensor
        #img = torch.randn_like(img) * 0.05 + img

        return img

