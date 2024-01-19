# -*- coding: utf-8 -*-

import re
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from matplotlib import cm
import scipy.io


# path integral data
# N_sample = 1000
mat = scipy.io.loadmat('../../data/observe_x_safety.mat')

obs_x = mat['observe_x']
obs_func = mat['observe_func']

# initial conditions and boundary conditions
ic = np.mgrid[1:4:0.05, 1:4:0.05, 0:0.1:1].reshape(3,-1).T
ic_func = np.ones([3600,1])

bc1 = np.mgrid[4:4.01:0.05, 1:4:0.05, 0:1:0.1].reshape(3,-1).T
bc1_func = np.zeros([600,1])

bc2 = np.mgrid[1:4:0.05, 4:4.01:0.05, 0:1:0.1].reshape(3,-1).T
bc2_func = np.zeros([600,1])


# training on T in [0, 0.5]
a = 0
b = 500

observe_x = np.append(obs_x[a:b], ic, axis=0)
observe_x = np.append(observe_x, bc1, axis=0)
observe_x = np.append(observe_x, bc2, axis=0)

observe_func = np.append(obs_func[a:b], ic_func, axis=0)
observe_func = np.append(observe_func, bc1_func, axis=0)
observe_func = np.append(observe_func, bc2_func, axis=0)


# PDE
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, j=2)
    dy_x1 = dde.grad.jacobian(y, x, j=0)
    dy_xx1 = dde.grad.hessian(y, x, i=0)
    dy_x2 = dde.grad.jacobian(y, x, j=1)
    dy_xx2 = dde.grad.hessian(y, x, i=1, j=1)
    # Backend tensorflow.compat.v1 or tensorflow
    return (
        dy_t
        - x[:, 0:1] * dy_x1
        - x[:, 1:2] * dy_x2
        - dy_xx1
        - 0.5 * dy_xx2
    )


observe_y = dde.icbc.PointSetBC(observe_x, observe_func, component=0)


# training setting
geom = dde.geometry.geometry_2d.Rectangle([1,1],[4,4])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_y],
    # num_domain=200,
    num_domain=600,
    # num_boundary=20,
    # num_initial=10,
    anchors=observe_x,
)

layer_size = [3] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, external_trainable_variables=[]
)
variable = dde.callbacks.VariableValue([], period=1000)

# PINN training
model.train(epochs=60000, callbacks=[variable])



# PINN testing
test_x = obs_x

yhat = model.predict(test_x)

difference = yhat - obs_func


# test on T = 1
a = 900
b = 1000

# Figure 7 in the appendix of the arxiv paper
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.scatter(obs_x[a:b,1], obs_x[a:b,0], obs_func[a:b]) # X: time, Y: state
ha.set_xlabel(r'$\xi_2$')
ha.set_ylabel(r'$\xi_1$')
ha.set_title("Path Integral MC")
plt.savefig("../../plots/path_integral_safety_probability_3d.pdf")

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.scatter(obs_x[a:b,1], obs_x[a:b,0], yhat[a:b]) # X: time, Y: state
ha.set_xlabel(r'$\xi_2$')
ha.set_ylabel(r'$\xi_1$')
ha.set_title("PINN Prediction")
plt.savefig("../../plots/PINN_prediction_safety_probability_3d.pdf")

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.scatter(obs_x[a:b,1], obs_x[a:b,0], difference[a:b]) # X: time, Y: state
ha.set_xlabel(r'$\xi_2$')
ha.set_ylabel(r'$\xi_1$')
ha.set_title("Difference")
plt.savefig("../../plots/difference_safety_probability_3d.pdf")