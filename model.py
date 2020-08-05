# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:49:34 2020

@author: seungjun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self,latent_dim):
        super(VAE, self).__init__()
        #for encoder
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc11 = nn.Linear(in_features=32*16*16, out_features=512)
        self.fc12 = nn.Linear(in_features=512, out_features=64)
        self.fc13 = nn.Linear(in_features=64, out_features=latent_dim)
        self.fc14 = nn.Linear(in_features=64, out_features=latent_dim)

        #for decoder
        self.fc21=nn.Linear(in_features=latent_dim, out_features=64)
        self.fc22=nn.Linear(in_features=64, out_features=512)
        self.fc23 = nn.Linear(in_features=512, out_features=32*16*16)
        self.conv5=nn.ConvTranspose2d(in_channels=32, out_channels=32,padding=1,output_padding=1,kernel_size=(3, 3),stride=2)
        self.conv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32,padding=1,kernel_size=(3, 3), stride=1)
        self.conv7=nn.ConvTranspose2d(in_channels=32, out_channels=3,padding=1,output_padding=1,kernel_size=(3, 3),stride=2)
        self.conv8=nn.ConvTranspose2d(in_channels=3, out_channels=1,padding=1,kernel_size=(3, 3), stride=1)

    def encoder(self, x):
        x= F.leaky_relu(self.conv1(x),0.2)
        x =F.leaky_relu(self.conv2(x),0.2)
        x =F.leaky_relu(self.conv3(x),0.2)
        x =F.leaky_relu(self.conv4(x),0.2)
        x = x.view(len(x),-1)
        x=F.leaky_relu(self.fc11(x),0.2)
        x = F.leaky_relu(self.fc12(x),0.2)
        mu=self.fc13(x)
        var=self.fc14(x)
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std,mu,var

    def decoder(self, z):
        m = F.leaky_relu(self.fc21(z),0.2)
        m = F.leaky_relu(self.fc22(m),0.2)
        m = F.leaky_relu(self.fc23(m),0.2)
        m = m.view(-1, 32,16,16)
        m =F.leaky_relu(self.conv5(m),0.2)
        m =F.leaky_relu(self.conv6(m),0.2)
        m =F.leaky_relu(self.conv7(m),0.2)
        m =F.leaky_relu(self.conv8(m),0.2)
        recon_x = F.sigmoid(m)
        return recon_x
    def forward(self, x):
        z,mu,var= self.encoder(x)
        return self.decoder(z), mu, var,z
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x)
        KLD = 0.5 * torch.sum( mu.pow(2) + logvar.exp() - logvar - 1)
        return (BCE + KLD)/x.shape[0]