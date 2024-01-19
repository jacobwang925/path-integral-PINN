#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 18 13:47:00 2023

@author: reecekeller
"""

from autoencoder_mlp import *
from utils import *
import numpy as np
import torch
import torch.optim as optim

input_dim = 3
output_dim = 1
latent_dim = 2
batchSize = 50 #number of samples per dimension
layer_vec = [64, latent_dim, 64]

# principled data generation
# [0, 1]^3 state-space
x = np.linspace(0, 1, batchSize)
g = np.meshgrid(x, x, x)
x_mat = np.array(g).T.reshape(-1, 3)
X = torch.tensor(x_mat)
X = X.to(torch.float32)
X = X[torch.randperm(X.size()[0])] 

# random data generation 
X = torch.rand(10**6, input_dim)
X_train = X[0:800000, :]
X_test = X[800000:, :]

X_train.requires_grad = True
X_test.requires_grad = True

if __name__ == '__main__':
    AE = mlp(input_dim, latent_dim, output_dim, layer_vec)
                                                    
    optimizer = optim.Adam(AE.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
    bl=[100]
    W_RC = 1
    w_CT = 10
    n_epochs = 5
    batch_size = 1000
    valid_loss = []; train_loss = []
    batches_per_epoch = X_train.size()[0] // batch_size
    preimage_frequency = 100
    delay = False
    
    print('batch_size: ', batch_size,
          '\nbatches_per_epoch: ', batches_per_epoch,
          '\npreimage_iterations: ', preimage_frequency,
          '\nnum_preimages_per_epoch: ', batches_per_epoch/preimage_frequency)
    AE.train()
    if delay:
        for epoch in range(n_epochs):
          for i in range(batches_per_epoch):
            start = i*batch_size
            Xbatch = X_train[start:start + batch_size]
            Xtest = X_test[start:start + batch_size]
        
            optimizer.zero_grad()
            #idx = torch.randperm(X.nelement())
            #X_train = X.view(-1)[idx].view(X.size())
            outputs, latents = AE(Xbatch)
            if (i % preimage_frequency == 0 or i==0):
              buffer_map1, buffer_map2 = mapBuilder(Xbatch, AE)
        
            l1 = encoderLoss(Xbatch, outputs)
            l2 = latentLossDelay(Xbatch, latents, buffer_map1, buffer_map2)
            loss = l1 + w_CT*l2
            #Validation loss
            outputsT, latentsT = AE(Xtest)
            test_buffer_map1, test_buffer_map2 = mapBuilder(X_test, AE)
            l1t = encoderLoss(Xtest, outputsT)
            l2t = latentLoss(Xtest, latentsT, buffer_map1, buffer_map2)
            loss_t = l1t + l2t
            valid_loss.append(loss_t)
        
        
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
        
            #scheduler.step()
            #best_loss = min(bl)
            #bl.append(loss.item())
        
            #if loss.item() < best_loss:
            #    torch.save(AE.state_dict(), '/content/drive/My Drive/Colab Notebooks/econfig8/run4/epoch' + str(epoch+1) + '_iter' + str(i) + '.pth')
        
            print(f'epoch: {epoch + 1}, iter: {i}, validation loss: {loss_t.item():.3f}, encoder loss: {l1.item():0.3f}, CT loss: {l2.item()}')
    else:
        # run standard training loop
        for epoch in range(n_epochs):
          for i in range(batches_per_epoch):
            start = i*batch_size
            Xbatch = X_train[start:start + batch_size]
            Xtest = X_test[start:start + batch_size]
        
            optimizer.zero_grad()
            outputs, latents = AE(Xbatch)
            map1, map2 = mapBuilder(Xbatch, AE)
            l1 = encoderLoss(Xbatch, outputs)
            l2 = latentLoss(Xbatch, latents, map1, map2)
            loss = l1 + w_CT*l2
            # Validation loss
            outputsT, latentsT = AE(Xtest)
            test_buffer_map1, test_buffer_map2 = mapBuilder(X_test, AE)
            l1t = encoderLoss(Xtest, outputsT)
            l2t = latentLoss(Xtest, latentsT, buffer_map1, buffer_map2)
            loss_t = l1t + l2t
            valid_loss.append(loss_t)
        
        
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
        
            #scheduler.step()
            #best_loss = min(bl)
            #bl.append(loss.item())
        
            #if loss.item() < best_loss:
            #    torch.save(AE.state_dict(), '/content/drive/My Drive/Colab Notebooks/econfig8/run4/epoch' + str(epoch+1) + '_iter' + str(i) + '.pth')
    
            print(f'epoch: {epoch + 1}, iter: {i}, validation loss: {loss.item():.3f}, encoder loss: {l1.item():0.3f}, CT loss: {l2.item()}')
