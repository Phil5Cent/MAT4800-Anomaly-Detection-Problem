import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from diffusers import AutoencoderKL



# === VAE Wrapper ===
class Anomaly_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = vae.encoder
        # self.decoder = vae.decoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

        # Freeze base VAE weights
        for param in self.vae.parameters():
            param.requires_grad = False

        self.tweak = AnomalyPusher().to(device)
        # self.vae.eval()

    def forward(self, x, label):
        # x should be normalized to [-1, 1] and resized to 512x512

        enc_out = self.vae.encode(x)
        latent_dist = enc_out["latent_dist"]
        latents = latent_dist.sample()
        latents[~label] = 0.6*latents[~label] + 0.4*self.tweak(latents[~label]) #fucks up the abnormal vectors

        # we essentially want the transformation to produce the identity for the good sample and fuck around for the bad sample. Think of it as an extra encoding step

        recon = self.vae.decode(latents)["sample"]
        # std = torch.exp(0.5 * latent_dist.logvar)
        return recon, latent_dist.mean, latent_dist.logvar

class AnomalyPusher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1)  # Output same shape
        )

    def forward(self, z):
        return self.conv_block(z)



# # === Basic Encoder ===
# class Anomaly_VAE_Encoder(nn.Module):
#     def __init__(self, latent_dim=100):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 4, 2, 1),  # 112x134 -> 56x67
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, 4, 2, 1),  # 56x67 -> 28x34
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1),  # 28x34 -> 14x16
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(128 * 16 * 14, latent_dim * 2)  # output: [mean | log_var]
#         )

#     def forward(self, x):
#         out = self.encoder(x)
#         mean, log_var = out.chunk(2, dim=1)
#         return mean, log_var


# # === Basic Decoder ===
# class Anomaly_VAE_Decoder(nn.Module):
#     def __init__(self, latent_dim=100):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 128 * 14 * 16),
#             nn.LeakyReLU(0.2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 14x16 -> 28x34
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 28x34 -> 56x68
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 3, 4, 2, 1),   # 56x68 -> 112x136
#             nn.Sigmoid()
#         )

#     def forward(self, z):
#         x = self.fc(z)
#         x = x.view(-1, 128, 14, 16)
#         return self.decoder(x)[..., :134]  # Crop width from 136 to 134


# # === VAE Wrapper ===
# class Anomaly_VAE(nn.Module):
#     def __init__(self, latent_dim=100):
#         super().__init__()
#         self.encoder = Anomaly_VAE_Encoder(latent_dim)
#         self.decoder = Anomaly_VAE_Decoder(latent_dim)

#     def reparameterize(self, mean, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def forward(self, x, label):
#         x = x[label] #only looking at normal data
#         mean, log_var = self.encoder(x)
#         z = self.reparameterize(mean, log_var)
#         recon = self.decoder(z)
#         return recon, mean, log_var




















































# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from base import BaseModel


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Anomaly_VAE(nn.Module):

#     def __init__(self, input_dim=(3*134*112), hidden_dim=400, latent_dim=100, device=device, latent_space_dim=20, mode='vector'):
#         super(Anomaly_VAE, self).__init__()

#         self.Encoder = Anomaly_VAE_Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device, latent_space_dim=latent_space_dim)
#         self.Decoder = Anomaly_VAE_Decoder(device=device, hidden_dim=hidden_dim, latent_space_dim=latent_space_dim)
#         self.mode = mode

        
#         # latent mean and variance 
#         self.mean_layer = nn.Linear(latent_dim, latent_space_dim) #transforms output vector logits into vector distribution mean information
#         self.log_var_layer = nn.Linear(latent_dim, latent_space_dim) #transforms output vector logits into vector distribution STD information

#     def reparameterization(self, mean, var): #generates a random vector from the sampled distribution
#         epsilon = torch.randn_like(var).to(device)      
#         z = mean + var*epsilon
#         return z


#     def forward(self, x, label):

#         x, x_intermediate = self.Encoder(x) #embeds vector

#         x_intermediate = x_intermediate[label]
        
#         x_normal = x[label]
#         # x_normal_avg = x_normal.mean(dim=0)
#         x_anomaly = x[~label]

#         mean_normal, log_var_normal = self.mean_layer(x_normal), self.log_var_layer(x_normal) #calculates mean/variance (distribution information) for the samples

#         mean_anom, log_var_anom = self.mean_layer(x_anomaly), self.log_var_layer(x_anomaly)





#         mean_normal_avg = mean_normal.mean(dim=0)

#         mean_distance = torch.norm(mean_anom - mean_normal_avg, dim=1)  # shape: [A]
#         mean_repulsion_loss = mean_distance.mean()  # scalar


#         logvar_normal_avg = log_var_normal.mean(dim=0)  # shape: [latent_space_dim]

#         logvar_distance = torch.norm(log_var_anom - logvar_normal_avg, dim=1)  # shape: [A]
#         logvar_repulsion_loss = logvar_distance.mean()  # scalar


#         #loss should not be defined here but I'm so over it whatever
#         dist_match_loss = -torch.log(mean_repulsion_loss + 0.2*logvar_repulsion_loss + 1e-6)

#         #recreation goal
#         z = self.reparameterization(mean_normal, log_var_normal) #generates a random vector from sampled distribution


#         z = self.Decoder(z, x_intermediate) #decodes vector generated from the same distribution


#         return dist_match_loss, mean_normal, log_var_normal, z


#         # #vector separation
#         # if self.mode == 'vector':

#         #     return x_anomaly, mean, log_var

#         # else:

#         # #recreation goal
#         #     z = self.reparameterization(mean, log_var) #generates a random vector from sampled distribution
            
#         #     x = self.Decoder(z) #decodes vector generated from the same distribution

#         #     return x


# #
# # def gaussian_nll(x, mean, var):
# #     return 0.5 * torch.sum(torch.log(2 * torch.pi * var) + ((x - mean) ** 2) / var)

# class Anomaly_VAE_Encoder(nn.Module):

#     def __init__(self, input_dim=(3*134*112), hidden_dim=200, latent_dim=50, device=device, latent_space_dim=8):
#         super(Anomaly_VAE_Encoder, self).__init__()

#         # MLP encoder

#         self.MLPstage1 = nn.Sequential(
#             nn.Linear(9984, hidden_dim),
#             nn.LeakyReLU(0.2))

#         self.MLPencoder = nn.Sequential(
            
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim//2, latent_dim),
#             nn.LeakyReLU(0.2)
#             )
        
#         # Convolutional Encoder
#         self.ConvEncoder = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=0),  
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=0),                           
#             nn.LeakyReLU(0.2),
#             )
        

        

#     def encode(self, x):
#         x = self.ConvEncoder(x) #convolves and compresses the image
#         x = x.view(x.size(0), -1) #Flattens for MLP
        
#         x = self.MLPstage1(x)
#         x_intermediate = x

#         x = self.MLPencoder(x) #processes the image, outputting vector representations


#         return x, x_intermediate




#     def forward(self, x):
#         x = self.encode(x) #gets distribution information
#         return x #x_hat, mean, log_var



# #gpt shortcut

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(channels),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(channels)
#         )

#     def forward(self, x):
#         return x + self.block(x)

# class Anomaly_VAE_Decoder(nn.Module):
#     def __init__(self, device=None, hidden_dim=200, latent_space_dim=8):
#         super(Anomaly_VAE_Decoder, self).__init__()

#         self.fc = nn.Sequential(
#             nn.Linear(latent_space_dim, 512 * 4 * 4),
#             nn.LeakyReLU(0.2)
#         )

#         self.skip_proj = nn.Linear(hidden_dim, 512 * 4 * 4)  # for skip connection

#         self.resblock = ResidualBlock(512)

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 → 8x8
#             # nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),

#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 → 16x16
#             # nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16 → 32x32
#             # nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),

#             nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # 32x32 → 64x64
#             nn.Sigmoid()  # Match [0, 1] output
#         )

#     def forward(self, z, x_intermediate):
#         x = self.fc(z)                          # [B, 512*4*4]
#         x_skip = self.skip_proj(x_intermediate)  # [B, 512*4*4]
#         x = x + x_skip                          # additive skip connection
#         x = x.view(-1, 512, 4, 4)               # reshape for conv

#         x = self.resblock(x)                   # Residual block
#         x = self.deconv(x)                     # Upsample
#         return x


# # #original
# # class Anomaly_VAE_Decoder(nn.Module):

# #     def __init__(self, device=device, hidden_dim=200 ,latent_space_dim=8):
# #         super(Anomaly_VAE_Decoder, self).__init__()
        
# #         # Decoder
# #         self.fc1 = nn.Sequential(
# #             nn.Linear(latent_space_dim, hidden_dim), #maps vectors from 9984 from size of convolutional output dimension. READ: maybe consider another layer here. This is pretty simplistic.
# #             nn.LeakyReLU(0.2),
# #             # nn.Linear(hidden_dim, 9984), #maps vectors from 9984 from size of convolutional output dimension. READ: maybe consider another layer here. This is pretty simplistic.
# #             # nn.LeakyReLU(0.2)
# #         )

# #         self.fc2 = nn.Sequential(
# #             nn.Linear(2*hidden_dim, 9984), #maps vectors from 9984 from size of convolutional output dimension. READ: maybe consider another layer here. This is pretty simplistic.
# #             nn.LeakyReLU(0.2),
# #             # nn.Linear(latent_space_dim, 9984), #maps vectors from 9984 from size of convolutional output dimension. READ: maybe consider another layer here. This is pretty simplistic.
# #             # nn.LeakyReLU(0.2)
# #         )


# #         self.deconv = nn.Sequential(
# #             nn.ConvTranspose2d(12, 6, kernel_size=3, stride=2),    # (12, 31, 26) → (6, 63, 53)
# #             nn.LeakyReLU(0.2),
# #             nn.ConvTranspose2d(6, 3, kernel_size=5, stride=2),     # (6, 63, 53) → (3, 133, 111)
# #             nn.Sigmoid()  # Final activation: match input range [0, 1]
# #         )




# #     def decode(self, z, x_intermediate):
        
# #         z = self.fc1(z)
# #         z = torch.cat((z, x_intermediate), dim=-1) #skip connection
# #         z = self.fc2(z)
# #         z = z.view(-1, 12, 32, 26) 
# #         z = self.deconv(z)
        
# #         return z


# #     def forward(self, z, x_intermediate):
# #         # x_intermediate=0 #NO skip connection currently active
# #         result = self.decode(z, x_intermediate) #uses trained model to produce a potential image from the same distribution
# #         return result #x_hat, mean, log_var
