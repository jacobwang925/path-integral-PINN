#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 18 11:40:58 2023

@author: reecekeller
"""

from typing import Union
import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        latent_size: int,
        output_size: int,
        size: list,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity',
) -> nn.Module:
    
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: list of elements=dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    input_layer = nn.Linear(input_size, size[0])
    output_layer = nn.Linear(size[-1], output_size)
    
    intermediate_layers = []
    
    for i in range(len(size)-1):
        intermediate_layers.extend([nn.Linear(size[i], size[i+1]), activation])

    model = nn.Sequential(
        input_layer,
        activation,
        *intermediate_layers,
        output_layer,
        output_activation
        )
    
    model.apply(init_weights)
        
    return model

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)