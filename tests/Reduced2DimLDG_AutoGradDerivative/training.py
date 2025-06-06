import torch
import torch.optim.adamax
from src.lossFunctions.Reduced2DimLDG import defDifONetLossPIML_w_AD
from src.architectures.DefDifONets.twoDimFeatureONet import two_dim_DefDifONet
from typing import Tuple
import matplotlib.pyplot as plt
import random
from pathlib import Path, PurePath
from tqdm import tqdm
import logging
from config import *
import dill as pickle

logger = logging.getLogger("logging_config")


def train(  model: two_dim_DefDifONet, x: torch.Tensor, numSolutions:int, epochs: int, solutionFeatures:list[torch.Tensor] =None, boundaryPoints: torch.Tensor = None,
            learningRate:float  = 1e-4,loadBestModel:bool = False, verbose:bool = True, showTrainingPlot:bool = True, modelName: str= "Two_Dim_DefDifONet",
            addRandomFeaturesToSolutions: bool = True, alpha:float = 1., beta:float = 0.1, delta:float = 1, deflationLossPoints: tuple[float,float] = (10000.,1.) , deflationCoefficient:float = 1., FrequencyReportLosses:int = 50)->Tuple[two_dim_DefDifONet, list[torch.Tensor]]:
    """This is the training funciton for the DifDefONet model for the reduced 2dim LDG model. It returns the trained model and the feature list containing the solution functions, which can be used with the trained model.

    Args:
        model (two_dim_DefDifONet): Model which we want to train.
        x (torch.Tensor): collocation points where we want to train the model.
        numSolutions (int): number of solutions which we want to obtain in the end
        epochs (int): Number of epochs during training.
        solutionFeatures (list[torch.Tensor], optional): Feature representation of the solution. If this is none, then it is initialized randomly. Defaults to None.
        boundaryPoints (torch.Tensor, optional): Boundary Points used for computing loss on boundary. If this is set to None, then no boundary loss will be computed. Defaults to None.
        learningRate (float, optional): Learnining Rate. Defaults to 1e-4.
        loadBestModel (bool, optional): Model with the lowest loss is constantly saved if this is True and in the end the model which had the lowest loss is returned.
                                        If this is set to true, then the model will be trained and the model obtained by the last epoch is saved and returned by this funciton. Defaults to False.
        verbose (bool, optional): Enables logs if set to true. Defaults to True.
        showTrainingPlot (bool, optional): Shows live losses during training in a graph. Defaults to True.
        modelName (str, optional): Name under which the model is saved. Defaults to "Laplacian_2D_DefDifONet".
        addRandomFeaturesToSolutions (bool, optional): Adds random noise to the feature representation of the solution funcitons. Defaults to True.
        alpha (float, optional): Weight coefficient for PDE loss. Defaults to 1..
        beta (float, optional): Weight coefficient for boundary loss. Defaults to 0.1.
        delta (float, optional): Weight coefficient for deflation loss. Defaults to 1..
        deflationLossPoints (tuple[float,float], optional): Points adjusting the linear deflation loss function. This contains (maxLoss, maxDistance). Defaults to (10000.,1.).
        deflationCoefficient (float, optional): Coefficient to adjust deflation loss. Defaults to 1..
        FrequencyReportLosses (int, optional): _description_. Defaults to 50.

    Returns:
        Tuple[Laplacian_2D_DefDifONet, list[torch.Tensor]]: Returns trained model and feature representation of the solution funcitons.
    """    

    pathSolutionFeatures = PurePath(pathData, "Reduced2DimLDG_AutoGradDerivative/SolutionFeatures/features.pkl")
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    bestLoss = torch.inf
    numFeaturesSolutions = model.numBranchFeatures *2 # times 2 is because of the number of outputs we have
    if solutionFeatures == None:
        listU = [ torch.rand((1,numFeaturesSolutions))  for i in range(numSolutions)]
    else:
        listU = solutionFeatures

    useBoundaryLossTerm = False
    if boundaryPoints != None:
        useBoundaryLossTerm = True

    #this is only used if addRandomFeaturesToSolutions is True
    meanForRandomFeatures = torch.zeros(1,numFeaturesSolutions)
    stdForRandomFeatures = [learningRate*i*10 for i in range( 10 )]

    #plotting settings:
    if showTrainingPlot:
        plt.ion()
        fig, ax = plt.subplots()
        lossesForPlot = []
        xValueForPlot = [FrequencyReportLosses * i for i in range(epochs // FrequencyReportLosses)]
        ax.set_ylim(0, 10)
        ax.set_xlim(0, epochs)
 
        ax.set_title("Live Loss Plot Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        line, = ax.plot(xValueForPlot[:len(lossesForPlot)], lossesForPlot)

    for epoch in tqdm(range(epochs)):
        if epoch == 1000:
            pass
        optimizer.zero_grad()
        
        # detach feature representation of funcitons
        for idxListU, u in enumerate(listU):
            #add some random noise to feature representation of functions
            if addRandomFeaturesToSolutions:
                randIdx = random.randint(0,9)
                stdThisItter = stdForRandomFeatures[randIdx]
                u = u + torch.normal(mean = meanForRandomFeatures,  std = stdThisItter)
            listU[idxListU] = u.detach()

        #compute loss
        modelOut = model(listU, x)
        modelOutBoundary = None
        if useBoundaryLossTerm:
            modelOutBoundary = model(listU, boundaryPoints)
        loss = defDifONetLossPIML_w_AD(  x = x, modelOut = modelOut, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary,
                                    eps = 0.02,deflationLossPoints = deflationLossPoints, deflationLossCoeff=deflationCoefficient, alpha = alpha, beta = beta, delta = delta)

        #update features of solution funciton representation
        for k in range(numSolutions):
            listU[k] = modelOut["features"][k]

        #update best loss so far
        if loss.item() < bestLoss:
            bestLoss = loss.item()
            #save model; pathTrainedModels is defined in config
            if loadBestModel:
                #torch.save(model, PurePath(pathTrainedModels, modelName +".pt"))
                with open(PurePath(pathTrainedModels, modelName +".pkl"), "wb") as f:
                    pickle.dump(model, f)
                with open(pathSolutionFeatures , "wb") as f:
                    pickle.dump(listU, f)

        loss.backward(retain_graph=True)
        optimizer.step()

        #some verbose stuff
        if epoch%FrequencyReportLosses == 0:
            if verbose:
                logger.info(f"epoch {epoch};     loss............{loss.item()}")

            #plotting
            if showTrainingPlot:
                lossesForPlot.append(loss.item())
            
                ax.set_ylim(0, max(lossesForPlot))
                ax.set_xlim(0,epoch)
                line.set_ydata(lossesForPlot)
                line.set_xdata(xValueForPlot[:len(lossesForPlot)])

                # Redraw the plot
                fig.canvas.draw()
                fig.canvas.flush_events()


        epoch += 1

    if loadBestModel:
        #load best model; pathTrainedModels is defined in config
        #model = torch.load(PurePath(pathTrainedModels, modelName +".pt")).to("cuda")
        with open(PurePath(pathTrainedModels, modelName +".pkl"), "rb") as f:
            model = pickle.load(f).to("cuda:0")

        with open(pathSolutionFeatures, "rb") as f:
            listU = pickle.load(f)
        for i in range(len(listU)):
            listU[i] = listU[i].to("cuda:0")
    else:
        #torch.save(model, PurePath(pathTrainedModels, modelName +".pt"))
        with open(PurePath(pathTrainedModels, modelName +".pkl"), "wb") as f:
            pickle.dump(model, f)

        
        with open(pathSolutionFeatures , "wb") as f:
            pickle.dump(listU, f)
            
    return model, listU