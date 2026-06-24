import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from attention_maps_dataset import AttentionMapsDataset
from vae import Autoencoder, ConvVAE, loss_function
from config import attn_maps_id_train_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data and create the data loader
my_transform = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR)

# ID data for training the autoencoder
id_dataset = AttentionMapsDataset(attn_maps_id_train_path, my_transform)

val_size = int(0.1 * len(id_dataset))
train_size = len(id_dataset) - val_size

train_set, val_set = random_split(id_dataset, [train_size, val_size])

id_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Create the autoencoder model
model = ConvVAE(latent_dim=128)

model = model.to(device)

num_epochs = 200

# Define the loss function (reconstruction error)
#criterion = torch.nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-5)

best_val_loss = float('inf')
patience_max = 5
patience_counter = patience_max

losses_per_epoch = []

# Train pipeline
for epoch in range(num_epochs):

    # ---TRAINING---
    model.train()

    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0

    for batch_idx, images in enumerate(id_loader):
        images = images.to(device)

        optimizer.zero_grad()

        outputs, mu, log_var = model(images)
        loss, recon_loss, kl_loss = loss_function(outputs, images, mu, log_var)

        loss.backward()

        optimizer.step()

        # Loss at batch level
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item() 
       
    scheduler.step()
    
    avg_loss = train_loss/ len(id_loader)
    avg_recon = train_recon / len(id_loader)
    avg_kl = train_kl / len(id_loader)

    losses_per_epoch.append(avg_loss)

    # ---VALIDATION---
    model.eval()

    val_loss = 0.0
    val_recon = 0.0
    val_kl = 0.0

    with torch.no_grad():
        for batch_idx, images in enumerate(val_loader):
            images = images.to(device)

            outputs, mu, log_var = model(images)
            loss, recon_loss, kl_loss = loss_function(outputs, images, mu, log_var)

            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_recon = val_recon / len(val_loader)
    avg_val_kl = val_kl / len(val_loader)
    #print(f"Epoch [{epoch+1}/{num_epochs}] Summary -> Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary")
    print(f"  Train -> Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
    print(f"  Val   -> Loss: {avg_val_loss:.4f} | Recon: {avg_val_recon:.4f} | KL: {avg_val_kl:.4f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = patience_max
        torch.save(model.state_dict(), "vae2.pth")  # Save the best model weights
    else:
        patience_counter -= 1
        if patience_counter == 0:
            print("Early stopping")
            break
