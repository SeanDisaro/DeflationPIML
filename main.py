import numpy as np
import torch
import deepxde as dde

center = torch.tensor([[0.,0.],[0.,1.],[0.,2.]]).view(3,-1)
radius = torch.tensor(1.)
geom = dde.geometry.geometry_2d.Disk([0.,0.], 1)

print(geom.random_points(100))

print(geom.boundary_constraint_factor(center, smoothness="C0+"))