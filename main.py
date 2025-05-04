import numpy as np
import torch
import deepxde as dde
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from logging_config import setup_logging
import logging

logger = setup_logging()

logger = logging.getLogger(__name__)

center = torch.tensor([[0.,0.],[0.,1.],[0.,2.]]).view(3,-1)
radius = torch.tensor(1.)
geom = dde.geometry.geometry_2d.Disk([0.,0.], 1)

points = torch.Tensor(geom.random_points(100))
points= points.to("cuda")
points.requires_grad = True
#print(geom.boundary_constraint_factor(center, smoothness="C0+"))

print(points.device)
model = Laplacian_2D_DefDifONet(                    
                    branch_layer =2 ,
                    branch_width = 200 ,
                    numBranchFeatures = 10 ,
                    trunk_layer =2 ,
                    trunk_width =200 ,
                    activationFunction =torch.tanh,
                    geom = geom,
                    DirichletHardConstraint = False,
                    skipConnection = False)

model.to("cuda")

U = [torch.rand((1,10*10)), torch.rand((1,10*10)), torch.rand((1,10*10))]

out = model(U,points)
out2 = model(out[-1],points)
print(out[-1])
print(out2[-1])
batchSize = points.shape[0]
outForAD = torch.concatenate(out[0])
batchSizeTimesNumInputfunc = batchSize *len(out[0]) 
out1_dxAD = torch.autograd.grad(outForAD, points[:,0].view(-1,1), torch.ones((batchSizeTimesNumInputfunc, 1), requires_grad = True).to("cuda"), allow_unused=True, create_graph=True)[0]


pass