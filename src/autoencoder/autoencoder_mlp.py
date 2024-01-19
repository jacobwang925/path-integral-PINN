#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 18 11:32:58 2023

@author: reecekeller
"""

from utils import *
from pytorch_utils import *
import torch.nn as nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, layer_vec):
        super(mlp, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layer_vec = layer_vec
        
        self.features = build_mlp(self.input_dim, self.latent_dim, self.output_dim, self.layer_vec)

    def forward(self, x):
        latent_layer = self.features[:(self.layer_vec.index(self.latent_dim)+1)*2](x)
        output_layer = self.features(x)
        return output_layer, latent_layer

#class CustomLoss(nn.Module):
#    def __init__(self):
#        super(CustomLoss, self).__init__()

#    def forward(self, input, output, latents):
      #with torch.autograd.set_detect_anomaly(True):
#      return encoderLoss(input, output), latentLoss(input, latents)


