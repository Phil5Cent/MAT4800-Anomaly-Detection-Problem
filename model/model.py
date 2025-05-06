import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# Screw that. All my homies HATE the MnistModel

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# Code from https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f

#Supporting Code that may be removed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Expected input data: batch size x channels x image height x image width

class Anomaly_VAE(nn.Module):

    def __init__(self, input_dim=(3*28*34), hidden_dim=200, latent_dim=50, device=device, latent_space_dim=20): #added latent_space_dim
        super(Anomaly_VAE, self).__init__()

        # standard encoder
        self.MLPencoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, latent_space_dim) #I cranked this number waaay up LOL
        self.log_var_layer = nn.Linear(latent_dim, latent_space_dim)
        
        # decoder
        self.decoder = nn.Sequential( #essentially this samples from the distribution within the latent space and uses the model to generate a new output.
            nn.Linear(latent_space_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() #Sigmoid is... questionable, particularly depending on data normalization techniques used.
            )
     
    def encode(self, x):

        x = x.reshape(x.size(0), -1) # flattens image into long tensor for MLP application
        x = self.MLPencoder(x) #uses base MLP based linear encoding
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        return mean, log_var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
