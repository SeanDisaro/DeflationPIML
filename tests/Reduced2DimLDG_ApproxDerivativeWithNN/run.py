import torch
import deepxde as dde
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from tests.Reduced2DimLDG_ApproxDerivativeWithNN.training import train
from src.lossFunctions.Reduced2DimLDG import boundaryFunctionExtension
from tests.Reduced2DimLDG_ApproxDerivativeWithNN.testing import plot_Q11_Q12, plot_nematic_director
import matplotlib.pyplot as plt



def run():
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])
    eps = 0.02
    safetyRadius = 0.1
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geom,dde.geometry.geometry_2d.Disk([0.,0.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,0.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([0.,1.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,1.], safetyRadius))
    geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Rectangle([0.+2*eps,0.+2*eps], [1.-2*eps,1.-2*eps]))
    geomWithoutSingularities = dde.geometry.csg.CSGUnion(geomWithoutSingularities, dde.geometry.geometry_2d.Rectangle([0.+4*eps,0.+4*eps], [1.-4*eps,1.-4*eps]))
    
    points = torch.Tensor(geomWithoutSingularities.random_points(1000))
    #boundaryPoints = torch.Tensor(geom.random_boundary_points(2000))
    #plt.scatter(points[:,0].view(-1).detach().cpu().numpy(), points[:,1].view(-1).detach().cpu().numpy(), s=0.1)
    #plt.show()

    points= points.to("cuda")

    #boundaryPoints = boundaryPoints.to("cuda")
    points.requires_grad = True
    numSolutions = 1
    learningRate = 5e-4
    #decreaseLearningRateEpoch = learningRate/2
    alpha = 1.
    beta = 1.
    gamma = 1.
    delta = 1.
    deflationCoefficient = 1.
    epochs= 1000
    deflationLossPoints = (1000.,0.2)
    
    model = Laplacian_2D_DefDifONet(                    
                    branch_layer = 1 ,
                    branch_width = 10000,
                    numBranchFeatures = 10 ,
                    trunk_layer = 1 ,
                    trunk_width = 10000 ,
                    activationFunction =torch.nn.Tanh(),
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = True,
                    DirichletConditionFunc1= lambda x: boundaryFunctionExtension(x, 3*eps),
                    DirichletConditionFunc2= lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
    )

	
    featuresSolution = [torch.rand((1,model.numBranchFeatures *10)) for i in range(numSolutions)]

    saveNameQ11Q12 = "Reduced2DimLDG_Results_" + str("zero")
    plot_Q11_Q12(model, featuresSolution, 40, saveName= saveNameQ11Q12)

    saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str("zero")
    plot_nematic_director(model, featuresSolution, 40, saveName= saveNamenematic_director)

    for k in range(3):
        if k == 1:
            pass  
        model,featuresSolution = train( model= model, x = points,  numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
        learningRate = learningRate /(10**k),loadBestModel = True, verbose = True, showTrainingPlot = True, 
        addRandomFeaturesToSolutions = False, alpha = alpha, beta = beta, gamma = gamma, delta = delta,deflationLossPoints=deflationLossPoints,deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)
        saveNameQ11Q12 = "Reduced2DimLDG_Results_" +  str(k)
        plot_Q11_Q12(model, featuresSolution, 40, saveName= saveNameQ11Q12)

        saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str(k)
        plot_nematic_director(model, featuresSolution, 40, saveName= saveNamenematic_director)



    return 0





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

