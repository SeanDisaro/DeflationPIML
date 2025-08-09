import torch
import deepxde as dde
from src.architectures.DefDifONets.twoDimFixedFeatureONet import two_dim_DefDifONet
from tests.Reduced2DimLDG_FixedBranchFeatures.training import train
from src.lossFunctions.Reduced2DimLDG import boundaryFunctionExtension
from tests.Reduced2DimLDG_FixedBranchFeatures.testing import plot_Q11_Q12, plot_nematic_director, plot_piml_Error
import matplotlib.pyplot as plt
from starDomainExtrapolation.starDomain import *



#####################################################################################################################
#These are some functions used for the boundary hard constraint
def boundaryConditionSpherical(starDomain, angles, boundaryFunction):
    return boundaryFunction(starDomain.getCartesianCoordinates( starDomain.radiusDomainFunciton(angles) ,angles))
  
def zeroOnBoundaryExtension(starDomain, input):
    radius, angles = starDomain.getSphericalCoordinates(input)
    return squaredRadial( radius / starDomain.radiusDomainFunciton(angles)).view(-1,1)
  
def DCBoundaryExtension(starDomain,input, boundaryFunction):
    radius, angles = starDomain.getSphericalCoordinates(input)
    return boundaryConditionSpherical(starDomain, angles, boundaryFunction) *  (1- squaredRadial( radius / starDomain.radiusDomainFunciton(angles))).view(-1,1) 


def squaredRadial(x):
    return (1 - x*x).view(-1,1)

def linearRadial(x):
    return (1 - x).view(-1,1)

def bfunc(x):
    return boundaryFunctionExtension(torch.cat((x[0],x[1]), dim=1), d= 0.06)
########################################################################################################################


def run():
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])

    squareAsStarDom = HyperCuboid(2, torch.tensor([0.5,0.5]), torch.tensor([1.,1.]))
    squareAsStarDom.updateDevice("cuda:0")

    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geom,dde.geometry.geometry_2d.Disk([0.,0.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,0.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([0.,1.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Disk([1.,1.], safetyRadius))
    # geomWithoutSingularities = dde.geometry.csg.CSGDifference(geomWithoutSingularities,dde.geometry.geometry_2d.Rectangle([0.+2*eps,0.+2*eps], [1.-2*eps,1.-2*eps]))
    # geomWithoutSingularities = dde.geometry.csg.CSGUnion(geomWithoutSingularities, dde.geometry.geometry_2d.Rectangle([0.+4*eps,0.+4*eps], [1.-4*eps,1.-4*eps]))
    
    # geomAroundCritical = dde.geometry.csg.CSGDifference(dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.]),
    #                                                         dde.geometry.geometry_2d.Rectangle([0.+4*eps,0.+4*eps],[1.-4*eps,1.-4*eps]))




    n = 65 # use even number for grid. Otherwise you get problems with differentiation, because you will include (0.5,0.5) in your set of points and the boundary extension function is not differentiable there.
    # Generate linearly spaced points between 0 and 1
    x = torch.linspace(0+0.001, 1-0.001, n) # added some safety space to border
    y = torch.linspace(0+0.001, 1-0.001, n)

    # Create a meshgrid
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # 'ij' indexing for matrix indexing

    # Stack and reshape to get [n, 2] shape
    points = torch.stack([xx, yy], dim=-1).reshape(-1, 2)


    # extraPoints = torch.Tensor(geomAroundCritical.random_points(1000))
    # points = torch.Tensor(geom.random_points(3000))
    # points = torch.concat((extraPoints,points))
    # boundaryPoints = torch.Tensor(geom.random_boundary_points(1000))
    # plt.scatter(points[:,0].view(-1).detach().cpu().numpy(), points[:,1].view(-1).detach().cpu().numpy(), s=0.1, c = "blue")
    # plt.scatter(boundaryPoints[:,0].view(-1).detach().cpu().numpy(), boundaryPoints[:,1].view(-1).detach().cpu().numpy(), s=0.1, c = "red")
    # plt.show()




    points= points.to("cuda")

    # boundaryPoints = boundaryPoints.to("cuda")
    points.requires_grad = True
    numSolutions = 6
    learningRate = 1e-3 # 1e-3
    # decreaseLearningRateEpoch = learningRate/2
    alpha = 1. # 2.
    beta = 1.
    delta = 100. # 2.6
    deflationCoefficient = 1.
    epochs= 40000
    deflationLossPoints = (1.,0.4) # 500
    
    model = two_dim_DefDifONet(
                    numSolutions = numSolutions,
                    numBranchFeatures = 6, #6 
                    trunk_layer = 1,
                    trunk_width = 2000, #2200
                    activationFunction = torch.nn.Tanh(),
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = False,
                    DirichletConditionFunc1 = lambda input: DCBoundaryExtension(squareAsStarDom,[input[:,0].view(-1,1), input[:,1].view(-1,1)], bfunc),
                    DirichletConditionFunc2 = lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
                    )

	

    saveNameQ11Q12 = "Reduced2DimLDG_Results_" + str("zero")
    plot_Q11_Q12(model,  33, saveName= saveNameQ11Q12)

    saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str("zero")
    plot_nematic_director(model, 33, saveName= saveNamenematic_director)

    savePIMLErrors = "Reduced2DimLDG_PIML_Errors_" +   str("zero")
    plot_piml_Error(model, 33, showPlot = False, saveName= savePIMLErrors )

    for k in range(1):
        # extraPoints = torch.Tensor(geomAroundCritical.random_points(1000))
        # points = torch.Tensor(geom.random_points(2300))
        # points = torch.concat((extraPoints,points))
        # points= points.to("cuda")
        # points.requires_grad = True
        model= train( model= model, x = points, epochs= epochs, boundaryPoints= None,
        learningRate = learningRate ,loadBestModel = True, verbose = True, showTrainingPlot = True, 
         alpha = alpha, beta = beta, delta = delta,deflationLossPoints=deflationLossPoints,deflationCoefficient=deflationCoefficient, FrequencyReportLosses = 20)

        #learningRate = learningRate /2

        saveNameQ11Q12 = "Reduced2DimLDG_Results_" +  str(k)
        plot_Q11_Q12(model,  40, saveName= saveNameQ11Q12)

        saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str(k)
        plot_nematic_director(model, 40, saveName= saveNamenematic_director)

        savePIMLErrors = "Reduced2DimLDG_PIML_Errors" +  str(k)
        plot_piml_Error(model, 40, showPlot = False, saveName= savePIMLErrors )


    return 0

