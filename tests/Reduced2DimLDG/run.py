import torch
import deepxde as dde
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from tests.Reduced2DimLDG.training import train
from src.lossFunctions.Reduced2DimLDG import boundaryFunctionExtension
from tests.Reduced2DimLDG.testing import plotResults
import matplotlib.pyplot as plt
def run():
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])
    safetyRadius = 0.1
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geom,dde.geometry.geometry_2d.Disk([0.,0.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,0.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([0.,1.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,1.], safetyRadius))
    eps = 0.02
    points = torch.Tensor(geomWithoutSingularities.random_points(3000))
    #plt.scatter(points[:,0].view(-1).detach().cpu().numpy(), points[:,1].view(-1).detach().cpu().numpy(), s=0.1)
    #plt.show()

    points= points.to("cuda")
    points.requires_grad = True
    numSolutions = 1
    learningRate = 5e-4
    alpha = 1.
    gamma = 1.
    delta = 0.1
    deflationCoefficient = 2
    epochs= 2000

    model = Laplacian_2D_DefDifONet(                    
                    branch_layer = 1 ,
                    branch_width = 10000,
                    numBranchFeatures = 100 ,
                    trunk_layer = 1 ,
                    trunk_width = 10000 ,
                    activationFunction =torch.tanh,
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = True,
                    DirichletConditionFunc1= lambda x: boundaryFunctionExtension(x, 3*eps),
                    DirichletConditionFunc2= lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
                    )
    
    featuresSolution = []
    featuresSolution .append(torch.rand((1,model.numBranchFeatures *10)))
    plotResults(model, featuresSolution, 20)
    model,featuresSolution = train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta,deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)
    
    #for i in range(50):
    #    modelOut = model(featuresSolution, points)
    #    for k in range(numSolutions):
    #        featuresSolution[k] = modelOut[-1][k]
    plotResults(model, featuresSolution, 20)

    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate/10,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta,deflationCoefficient=deflationCoefficient , FrequencyReportLosses = 20)
    
    #for i in range(50):
    #    modelOut = model(featuresSolution, points)
    #    for k in range(numSolutions):
    #        featuresSolution[k] = modelOut[-1][k]
    
    plotResults(model, featuresSolution, 20)

    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate/100,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta,deflationCoefficient=deflationCoefficient , FrequencyReportLosses = 20)
    
    #for i in range(50):
    #    modelOut = model(featuresSolution, points)
    #    for k in range(numSolutions):
    #        featuresSolution[k] = modelOut[-1][k]
    
    plotResults(model, featuresSolution, 20)



    featuresSolution .append(torch.rand((1,model.numBranchFeatures *10)))



    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta, deflationCoefficient=deflationCoefficient , FrequencyReportLosses = 20)
    
    #for i in range(50):
    #    modelOut = model(featuresSolution, points)
    #    for k in range(numSolutions):
    #        featuresSolution[k] = modelOut[-1][k]
    
    plotResults(model, featuresSolution, 20)

    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
            learningRate = learningRate/10,loadBestModel = False, verbose = True, showTrainingPlot = True, 
            addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta, deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)
    
    #for i in range(50):
    #    modelOut = model(featuresSolution, points)
    #    for k in range(numSolutions):
    #        featuresSolution[k] = modelOut[-1][k]
    plotResults(model, featuresSolution, 20)

    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
    learningRate = learningRate/100,loadBestModel = False, verbose = True, showTrainingPlot = True, 
    addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta, deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)
    
    plotResults(model, featuresSolution, 20)

    model,featuresSolution =train( model= model, x = points, numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
    learningRate = learningRate/1000,loadBestModel = False, verbose = True, showTrainingPlot = True, 
    addRandomFeaturesToSolutions = False, alpha = alpha, beta = 0.1, gamma = gamma, delta = delta, deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)
    
    plotResults(model, featuresSolution, 20)