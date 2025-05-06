import torch
import deepxde as dde
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from tests.Reduced2DimLDG.training import train
from src.lossFunctions.Reduced2DimLDG import boundaryFunctionExtension

def run():
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])
    eps = 0.02
    points = torch.Tensor(geom.random_points(3000))
    points= points.to("cuda")
    points.requires_grad = True
    numSolutions = 6
    learningRate = 1e-4
    alpha = 0.1
    gamma = 1.
    delta = 10.
    model = Laplacian_2D_DefDifONet(                    
                    branch_layer = 1 ,
                    branch_width = 10000,
                    numBranchFeatures = 100 ,
                    trunk_layer = 1 ,
                    trunk_width = 8000 ,
                    activationFunction =torch.tanh,
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = False,
                    DirichletConditionFunc1= lambda x: boundaryFunctionExtension(x, 3*eps),
                    DirichletConditionFunc2= lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
                    )


    model,featuresSolution = train( model= model, x = points, numSolutions = numSolutions, epochs= 1000, boundaryPoints= None,
            learningRate = learningRate,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta, FrequencyReportLosses = 50)
    for i in range(50):
        modelOut = model(featuresSolution, points)
        for k in range(numSolutions):
            featuresSolution[k] = modelOut[-1][k]
    alpha = alpha * 10
    gamma =gamma  * 10
    delta = delta * 10
    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= 10000, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = 0.1, beta = 0.1, gamma = 1., delta = 1., FrequencyReportLosses = 50)
    for i in range(50):
        modelOut = model(featuresSolution, points)
        for k in range(numSolutions):
            featuresSolution[k] = modelOut[-1][k]
    alpha = alpha * 10
    gamma =gamma  * 10
    delta = delta * 10
    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= 10000, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = 0.1, beta = 0.1, gamma = 1., delta = 1., FrequencyReportLosses = 50)
    for i in range(50):
        modelOut = model(featuresSolution, points)
        for k in range(numSolutions):
            featuresSolution[k] = modelOut[-1][k]
    alpha = alpha * 10
    gamma = gamma  * 10
    delta = delta * 10
    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= 10000, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = 0.1, beta = 0.1, gamma = 1., delta = 1., FrequencyReportLosses = 50)