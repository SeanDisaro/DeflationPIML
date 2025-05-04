import torch
import deepxde as dde
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from tests.Reduced2DimLDG.training import train
from src.lossFunctions.Reduced2DimLDG import boundaryFunctionExtension

def run():
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])
    eps = 0.02
    points = torch.Tensor(geom.random_points(2000))
    points= points.to("cuda")
    points.requires_grad = True
    numSolutions = 6

    model = Laplacian_2D_DefDifONet(                    
                    branch_layer = 1 ,
                    branch_width = 2000 ,
                    numBranchFeatures = 200 ,
                    trunk_layer = 2 ,
                    trunk_width = 1000 ,
                    activationFunction =torch.tanh,
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = True,
                    DirichletConditionFunc1= lambda x: boundaryFunctionExtension(x, 3*eps),
                    DirichletConditionFunc2= lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
                    )


    train( model= model, x = points, numSolutions = numSolutions, epochs= 1000, boundaryPoints= None,
            learningRate = 1e-4,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = True, alpha = 1., beta = 0.1, gamma = 1., delta = 1, FrequencyReportLosses = 50)