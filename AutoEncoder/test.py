from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import numpy as np
import sys
import os
import torch.nn as nn 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import attn_maps_id_test_path, attn_maps_ood_path
from vae import ConvVAE, loss_function
from attention_maps_dataset import AttentionMapsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
])

# LOADING DATA
# ID test
#test_id_dataset = AttentionMapsDataset(attn_maps_id_test_path, my_transform)
#test_id_loader = DataLoader(test_id_dataset, batch_size=64, shuffle=False)

# OOD test
test_ood_dataset = AttentionMapsDataset("../attention_maps/ood/", my_transform)
test_ood_loader = DataLoader(test_ood_dataset, batch_size=64, shuffle=False)

# MODEL
trained_model = ConvVAE(latent_dim=256)

checkpoint = torch.load("vae.pth")
trained_model.load_state_dict(checkpoint)

trained_model = trained_model.to(device)
trained_model.eval()

all_errors = []

# Test pipeline
with torch.no_grad():
    for batch_idx, images in enumerate(test_ood_loader):
        images = images.to(device)

        outputs, mu, log_var = trained_model(images)
        loss = loss_function(outputs, images, mu, log_var, reconstruct_per_sample=True)

        all_errors.extend(loss.cpu().numpy())

np.save("ood_errors", np.array(all_errors))

