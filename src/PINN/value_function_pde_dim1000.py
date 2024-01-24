# -*- coding: utf-8 -*-

import re
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from matplotlib import cm
import scipy.io

# path integral training data
# N_sample = 1000
mat = scipy.io.loadmat('../../data/observe_x_1kd_MC_1000.mat')

obs_x = mat['observe_x']
obs_func = mat['observe_func']


# initial conditions
ic = np.mgrid[1:2:0.05, 1:2:0.05, 1.5:1.45:-0.1].reshape(3,-1).T
ic_func = np.exp(0.001*(-0.5*ic[:,0]**2 - 0.5*ic[:,1]**2)) # Q scaled by 0.001
ic_func = ic_func.reshape(-1,1)


# training on T in [1, 1.5]
a = 1000
b = 1500

observe_x = np.append(obs_x[a:b], ic, axis=0)
observe_func = np.append(obs_func[a:b], ic_func, axis=0)

# PDE
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, j=2)
    dy_x1 = dde.grad.jacobian(y, x, j=0)
    dy_xx1 = dde.grad.hessian(y, x, i=0)
    dy_x2 = dde.grad.jacobian(y, x, j=1)
    dy_xx2 = dde.grad.hessian(y, x, i=1, j=1)
    return (
        dy_t
        + x[:, 0:1] * dy_x1
        + x[:, 1:2] * dy_x2
        + 250 * dy_xx1
        + 250 * dy_xx2
        - 0.001*(0.5*x[:, 0:1]**2 + 0.5*x[:, 1:2]**2) * y
    )


observe_y = dde.icbc.PointSetBC(observe_x, observe_func, component=0)


# training setting
geom = dde.geometry.geometry_2d.Rectangle([1,1],[2,2])
timedomain = dde.geometry.TimeDomain(0, 1.5)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_y],
    num_domain=800,
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

# path integral MC data for comparison
# N_sample = 10000
mat0 = scipy.io.loadmat('../../data/observe_x_1kd_10e5.mat')

obs0_x = mat0['observe_x']
obs0_func = mat0['observe_func']

test_x = obs0_x

yhat = model.predict(test_x)

difference = yhat - obs0_func

# test on T = 0.5
a = 500
b = 600

# Figure 4 in the paper
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.scatter(obs0_x[a:b,1], obs0_x[a:b,0], obs_func[a:b]) # X: time, Y: state
ha.set_xlabel(r'$\xi_2$')
ha.set_ylabel(r'$\xi_1$')
ha.set_title("Path Integral MC")
plt.savefig("../../plots/path_integral_value_function_1000d.pdf")

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.scatter(obs0_x[a:b,1], obs0_x[a:b,0], yhat[a:b]) # X: time, Y: state
ha.set_xlabel(r'$\xi_2$')
ha.set_ylabel(r'$\xi_1$')
ha.set_title("PINN Prediction")
plt.savefig("../../plots/PINN_prediction_value_function_1000d.pdf")

# hf = plt.figure()
# ha = hf.add_subplot(111, projection='3d')
# ha.scatter(obs0_x[a:b,1], obs0_x[a:b,0], yhat[a:b]-obs_func[a:b]) # X: time, Y: state
# ha.set_xlabel(r'$\xi_2$')
# ha.set_ylabel(r'$\xi_1$')
# ha.set_title("Difference")
# plt.savefig("../../plots/difference_value_function_1000d.pdf")
# plt.show()