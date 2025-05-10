import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Anomaly_VAE(nn.Module):

    def __init__(self, input_dim=(3*134*112), hidden_dim=200, latent_dim=50, device=device, latent_space_dim=8):
        super(Anomaly_VAE, self).__init__()

        # MLP encoder
        self.MLPencoder = nn.Sequential(
            nn.Linear(9984, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # Convolutional Encoder
        self.ConvEncoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=0),  
            nn.LeakyReLU(0.2),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=0),                           
            nn.LeakyReLU(0.2),
            )
        

        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, latent_space_dim) #transforms output vector logits into vector distribution mean information
        self.log_var_layer = nn.Linear(latent_dim, latent_space_dim) #transforms output vector logits into vector distribution STD information
        
        # Decoder
        self.fc = nn.Sequential(
            nn.Linear(latent_space_dim, 9984), #maps vectors from 9984 from size of convolutional output dimension. READ: maybe consider another layer here. This is pretty simplistic.
            nn.LeakyReLU(0.2)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(12, 6, kernel_size=3, stride=2),    # (12, 31, 26) → (6, 63, 53)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(6, 3, kernel_size=5, stride=2),     # (6, 63, 53) → (3, 133, 111)
            nn.Sigmoid()  # Final activation: match input range [0, 1]
        )




    def decode(self, x):
        
        x = self.fc(x)
        x = x.view(-1, 12, 32, 26) 
        x = self.deconv(x)

        
        return x



    def encode(self, x):
        x = self.ConvEncoder(x) #convolves and compresses the image
        x = x.view(x.size(0), -1) #Flattens for MLP
        x = self.MLPencoder(x) #processes the image, outputting vector representations

        mean, log_var = self.mean_layer(x), self.log_var_layer(x) #calculates mean/variance (distribution information) for the samples
        return mean, log_var

    def reparameterization(self, mean, var): #generates a random vector from the sampled distribution
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z


    def forward(self, x):
        mean, log_var = self.encode(x) #gets distribution information
        z = self.reparameterization(mean, log_var) #generates a random vector from sampled distribution
        result = self.decode(z) #uses trained model to produce a potential image from the same distribution
        return result #x_hat, mean, log_var
