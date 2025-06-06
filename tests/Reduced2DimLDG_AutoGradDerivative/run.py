import torch
import deepxde as dde
from src.architectures.DefDifONets.twoDimFeatureONet import two_dim_DefDifONet
from tests.Reduced2DimLDG_AutoGradDerivative.training import train
from src.lossFunctions.Reduced2DimLDG import boundaryFunctionExtension
from tests.Reduced2DimLDG_AutoGradDerivative.testing import plot_Q11_Q12, plot_nematic_director
import matplotlib.pyplot as plt



def run():
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])
    eps = 0.02
    safetyRadius = 0.1

    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geom,dde.geometry.geometry_2d.Disk([0.,0.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,0.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([0.,1.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,1.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Rectangle([0.+2*eps,0.+2*eps], [1.-2*eps,1.-2*eps]))
    # geomWithoutSingularities = dde.geometry.csg.CSGUnion(geomWithoutSingularities, dde.geometry.geometry_2d.Rectangle([0.+4*eps,0.+4*eps], [1.-4*eps,1.-4*eps]))
    
    geomAroundCritical = dde.geometry.csg.CSGDifference(dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.]),
                                                            dde.geometry.geometry_2d.Rectangle([0.+4*eps,0.+4*eps],[1.-4*eps,1.-4*eps]))

    extraPoints = torch.Tensor(geomAroundCritical.random_points(500))
    points = torch.Tensor(geom.random_points(2000))
    points = torch.concat((extraPoints,points))
    # boundaryPoints = torch.Tensor(geom.random_boundary_points(1000))
    #plt.scatter(points[:,0].view(-1).detach().cpu().numpy(), points[:,1].view(-1).detach().cpu().numpy(), s=0.1, c = "blue")
    # plt.scatter(boundaryPoints[:,0].view(-1).detach().cpu().numpy(), boundaryPoints[:,1].view(-1).detach().cpu().numpy(), s=0.1, c = "red")
    # plt.show()

    points= points.to("cuda")

    # boundaryPoints = boundaryPoints.to("cuda")
    points.requires_grad = True
    numSolutions = 1
    learningRate = 1e-6
    #decreaseLearningRateEpoch = learningRate/2
    alpha = 1.
    beta = 1.
    delta = 1.
    deflationCoefficient = 1.
    epochs= 1000
    deflationLossPoints = (8000.,1.5)
    
    model = two_dim_DefDifONet(                    
                    branch_layer = 10,
                    branch_width = 1000,
                    numBranchFeatures = 100,
                    trunk_layer = 10,
                    trunk_width = 1000 ,
                    activationFunction =torch.nn.Tanh(),
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = True,
                    DirichletConditionFunc1 = lambda x: boundaryFunctionExtension(x, 3*eps),
                    DirichletConditionFunc2 = lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
    )

	
    featuresSolution = [torch.rand((1,model.numBranchFeatures *2)) for i in range(numSolutions)]

    saveNameQ11Q12 = "Reduced2DimLDG_Results_" + str("zero")
    plot_Q11_Q12(model, featuresSolution, 40, saveName= saveNameQ11Q12)

    saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str("zero")
    plot_nematic_director(model, featuresSolution, 40, saveName= saveNamenematic_director)

    for k in range(1):
        model,featuresSolution = train( model= model, x = points,  numSolutions = numSolutions, epochs= epochs, solutionFeatures= featuresSolution, boundaryPoints= None,
        learningRate = learningRate /(10**k),loadBestModel = True, verbose = True, showTrainingPlot = True, 
        addRandomFeaturesToSolutions = False, alpha = alpha, beta = beta, delta = delta,deflationLossPoints=deflationLossPoints,deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)


        saveNameQ11Q12 = "Reduced2DimLDG_Results_" +  str(k)
        plot_Q11_Q12(model, featuresSolution, 40, saveName= saveNameQ11Q12)

        saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str(k)
        plot_nematic_director(model, featuresSolution, 40, saveName= saveNamenematic_director)


    return 0

