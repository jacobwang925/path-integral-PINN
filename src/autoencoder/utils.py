#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 18 11:29:16 2023
@author: reecekeller
"""

import numpy as np
import torch
import deepxde as dde
from bisect import bisect_left



def mapBuilder(input, model):

  # initialize dictionaries
  map1 = {}
  map2 = {}

  _, latent = model(input)

  #define thresholds as 1/5th the avg. difference between elements
  thresh1 = 0.2*np.mean(np.abs(np.diff(latent[:, 0].detach().numpy())))
  thresh2 = 0.2*np.mean(np.abs(np.diff(latent[:, 1].detach().numpy())))
  for state in input:

    # retrieve latent activations
    _, latent = model(state)
    idx = torch.where(torch.all(input==state,axis=1))
    state_idx = idx[0].tolist()[0]

    # convert from tensor to numpy array
    state = state.detach().clone().numpy()
    latent = latent.detach().clone().numpy()

    # build x -> xi dictionary
    xi_1 = latent[0]
    xi_2 = latent[1]

    flag = 0

    for key in map1:
      if np.abs(xi_1 - float(key)) < thresh1:
        map1[key].append([state, state_idx])
        flag+=1

    if flag == 0 :
      map1.update({str(xi_1): [[state, state_idx]]})
    else:
      flag = 0

    for key in map2:
      if np.abs(xi_2 - float(key)) < thresh2:
        map2[key].append([state, state_idx])
        flag+=1

    if flag == 0 :
      map2.update({str(xi_2): [[state, state_idx]]})
    else:
      flag = 0

  return map1, map2

def get_idx(input, map):
  idx_dict = {}
  for xi in map:
    states = map[xi]
    for state in states:
      idx_dict.update({xi: state_idx})
  return idx_dict

def f(x):
  A = torch.tensor([[1., 0., 1.], [0., 1., -1.], [0., 0., 1.]]) # system parameters  #= torch.randint(-1, 1, (3, 3), dtype=torch.float32)
  return torch.matmul(A, torch.transpose(torch.tensor(np.reshape(x, (1, 3))), 0, 1))

def encoderLoss(input, output):
  # assumes function args are tensors
  state_cost = 0.5*(input[:, 0] + input[:, 1])**2 + 0.5*input[:, 2]**2
  return torch.mean((state_cost - output)**2)

def latentLoss(input, latents, map1, map2):
  ct = []
  ct_loss = []
  for xi_1 in map1:
    A_x = []
    B_x = []
    states = map1[xi_1]
    for s in states:
      state = s[0]
      state_idx = s[1]

      J = dde.grad.jacobian(latents, input, i=0)
      a_x = torch.norm(J[state_idx, :])**2 + 1e-4
      A_x.append(a_x)

      H11 = dde.grad.hessian(latents, input, component = 0, i = 0, j = 0)
      H22 = dde.grad.hessian(latents, input, component = 0, i = 1, j = 1)
      H33 = dde.grad.hessian(latents, input, component = 0, i = 2, j = 2)
      b_x = (torch.matmul(J[state_idx, :], f(state)) \
       + 0.5*(H11[state_idx] + H22[state_idx] + H33[state_idx]))/a_x
      B_x.append(b_x)

    ct.append((1/len(states))*((max(A_x)-min(A_x))**2 + (max(B_x)-min(B_x))**2))
  #print(ct)
  ct_loss.append(torch.mean(torch.tensor(ct)))
  ct = []

  for xi_2 in map2:
    A_x = []
    B_x = []
    states = map2[xi_2]
    for s in states:
      state = s[0]
      state_idx = s[1]

      J = dde.grad.jacobian(latents, input, i=1)
      a_x = torch.norm(J[state_idx, :])**2 + 1e-4
      A_x.append(a_x)

      H11 = dde.grad.hessian(latents, input, component = 1, i = 0, j = 0)
      H22 = dde.grad.hessian(latents, input, component = 1, i = 1, j = 1)
      H33 = dde.grad.hessian(latents, input, component = 1, i = 2, j = 2)
      b_x = (torch.matmul(J[state_idx, :], f(state)) \
       + 0.5*(H11[state_idx] + H22[state_idx] + H33[state_idx]))/a_x
      B_x.append(b_x)

    ct.append((1/len(states))*((max(A_x)-min(A_x))**2 + (max(B_x)-min(B_x))**2))

  ct_loss.append(torch.mean(torch.tensor(ct)))

  return torch.mean(torch.tensor(ct_loss))

def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def latentLossDelay(input, latents, map1, map2):
  ct_loss = []

  ct_temp = []
  sorted_map = sorted(list(np.float_(list(map1.keys()))))
  for xi_1 in latents[:, 0]:
    A_x = []
    B_x = []
    if str(xi_1) in map1:
      states = map1[str(xi_1)]
    else:
      neighbor = take_closest(sorted_map, xi_1.float())
      states = map1[str(neighbor)]
    for s in states:
      state = s[0]
      state_idx = s[1]

      J = dde.grad.jacobian(latents, input, i=0)
      a_x = torch.norm(J[state_idx, :])**2+0.0001
      A_x.append(a_x)

      H11 = dde.grad.hessian(latents, input, component = 0, i = 0, j = 0)
      H22 = dde.grad.hessian(latents, input, component = 0, i = 1, j = 1)
      H33 = dde.grad.hessian(latents, input, component = 0, i = 2, j = 2)
      b_x = (torch.matmul(J[state_idx, :], f(state)) \
       + 0.5*(H11[state_idx] + H22[state_idx] + H33[state_idx]))/a_x
      B_x.append(b_x)

    ct_temp.append((1/len(states))*((max(A_x)-min(A_x))**2 + (max(B_x)-min(B_x))**2))
  ct_loss.append(torch.mean(torch.tensor(ct_temp)))

  ct_temp = []
  sorted_map = sorted(list(np.float_(list(map2.keys()))))
  for xi_2 in latents[:, 1]:
    A_x = []
    B_x = []
    if str(xi_2) in map2:
      states = map2[str(xi_2)]
    else:
      neighbor = take_closest(sorted_map, xi_2.float())
      states = map2[str(neighbor)]
    for s in states:
      state = s[0]
      state_idx = s[1]

      J = dde.grad.jacobian(latents, input, i=1)
      a_x = torch.norm(J[state_idx, :])**2 + 1e-5
      A_x.append(a_x)

      H11 = dde.grad.hessian(latents, input, component = 1, i = 0, j = 0)
      H22 = dde.grad.hessian(latents, input, component = 1, i = 1, j = 1)
      H33 = dde.grad.hessian(latents, input, component = 1, i = 2, j = 2)
      b_x = (torch.matmul(J[state_idx, :], f(state)) \
       + 0.5*(H11[state_idx] + H22[state_idx] + H33[state_idx]))/a_x
      B_x.append(b_x)

    ct_temp.append((1/len(states))*((max(A_x)-min(A_x))**2 + (max(B_x)-min(B_x))**2))
  ct_loss.append(torch.mean(torch.tensor(ct_temp)))

  return torch.mean(torch.tensor(ct_loss))