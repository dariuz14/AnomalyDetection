from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform

        for root, _, files in os.walk(data_dir):
            for f in files:
                self.data.append(os.path.join(root, f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.data[idx]).convert('RGB') # RGB for original images
        except OSError:
            print(f"Corrupted image: {self.data[idx]}")
            
        if self.transform:
            img = self.transform(img)
            
        return img
