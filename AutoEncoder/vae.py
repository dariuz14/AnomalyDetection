import torch.nn as nn
import torch

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        
        # Flatten: 256*8*8 = 16384
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # Da latent → decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)  # unflatten
        return self.decoder(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

def loss_function(x_reconstructed, x, mu, log_var, beta=0.05, reconstruct_per_sample=False):
   
    # Ricostruzione: media su tutti i pixel e batch
    bce = nn.functional.mse_loss(x_reconstructed.view(x.size(0), -1), x.view(x.size(0), -1), reduction='none')
    recon_loss_per_sample = torch.sum(bce, dim=1)
    recon_loss = torch.mean(recon_loss_per_sample)

    # KL Divergence: somma sulla dimensione latente, media sulla batch
    sum_terms = 1 + log_var - mu.pow(2) - log_var.exp()
    kl_batch = -0.5 * torch.sum(sum_terms, dim=1)
    kl_loss = torch.mean(kl_batch)
    
    total_loss = recon_loss + beta * kl_loss

    if reconstruct_per_sample:
        return recon_loss_per_sample

    return total_loss, recon_loss, kl_loss
