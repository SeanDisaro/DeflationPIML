import torch
from src.lossFunctions.Reduced2DimLDG import defDifONetLossPIML
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from typing import Tuple
import matplotlib.pyplot as plt
import random
from pathlib import Path, PurePath
from tqdm import tqdm
import logging
from config import *



logger = logging.getLogger(__name__)

def train(  model: Laplacian_2D_DefDifONet, x: torch.Tensor, numSolutions:int, epochs: int, boundaryPoints: torch.Tensor = None,
            learningRate:float  = 1e-4,loadBestModel:bool = False, verbose:bool = True, showTrainingPlot:bool = True, modelName: str= "Laplacian_2D_DefDifONet",
            addRandomFeaturesToSolutions: bool = True, alpha:float = 1., beta:float = 0.1, gamma:float = 1., delta:float = 1, FrequencyReportLosses:int = 50)->Tuple[Laplacian_2D_DefDifONet, list[torch.Tensor]]:
    """This is the training funciton for the DifDefONet model for the reduced 2dim LDG model. It returns the trained model and the feature list containing the solution functions, which can be used with the trained model.

    Args:
        model (Laplacian_2D_DefDifONet): Model which we want to train.
        x (torch.Tensor): collocation points where we want to train the model.
        numSolutions (int): number of solutions which we want to obtain in the end
        epochs (int): Number of epochs during training.
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
        gamma (float, optional): Weight coefficient for derivative loss. Defaults to 1..
        dleta (float, optional): Weight coefficient for deflation loss. Defaults to 1..
        FrequencyReportLosses (int, optional): _description_. Defaults to 50.

    Returns:
        Tuple[Laplacian_2D_DefDifONet, list[torch.Tensor]]: Returns trained model and feature representation of the solution funcitons.
    """    


    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    bestLoss = torch.inf
    numFeaturesSolutions = model.numBranchFeatures *10 # times 10 is because of the number of outputs we have, i.e. the derivatives and the dimesnion of the solution (which is 2 here)
    listU = [ torch.rand((1,numFeaturesSolutions))  for i in range(numSolutions)]

    useBoundaryLossTerm = False
    if boundaryPoints != None:
        useBoundaryLossTerm = True

    #this is only used if addRandomFeaturesToSolutions is True
    meanForRandomFeatures = torch.zeros(1,numFeaturesSolutions)
    stdForRandomFeatures = [learningRate*i*10 for i in range( 10 )]
    lossesForPlot = []
    xValueForPlot = [FrequencyReportLosses * i for i in range(epochs // FrequencyReportLosses)]
    #plotting settings:
    if showTrainingPlot:
        plt.ion()
        fig, ax = plt.subplots()
        x = []
        y = []
        ax.set_ylim(0, 10)
        ax.set_xlim(0, epochs)
        line, = ax.plot(x, y)
        ax.set_title("Live Loss Plot Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        #add some random noise to feature representation of functions
        if addRandomFeaturesToSolutions:
            for idxListU, u in enumerate(listU):
                randIdx = random.randint(0,10)
                stdThisEpoch = stdForRandomFeatures[randIdx]
                u = u + torch.normal(mean = meanForRandomFeatures,  std = stdThisEpoch, device = u.device)
                listU[idxListU] = u

        #compute loss
        modelOut = model(listU, x)
        modelOutBoundary = None
        if useBoundaryLossTerm:
            modelOutBoundary = model(listU, x)
        loss = defDifONetLossPIML(  x = x, modelOut = modelOut, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary,
                                    eps = 0.02, deflationLossCoeff=1, alpha = alpha, beta = beta, gamma = gamma, delta = delta)

        #update features of solution funciton representation
        for k in range(numSolutions):
            listU[k] = modelOut[-1][k]

        #update best loss so far
        if loss.item() < bestLoss:
            bestLoss = loss.item()
            #save model; pathTrainedModels is defined in config
            if loadBestModel:
                torch.save(model, PurePath(pathTrainedModels, modelName +".pt"))
        
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
            line.set_ydata(y)
            line.set_xdata(xValueForPlot[:len(lossesForPlot)])

            # Redraw the plot
            fig.canvas.draw()
            fig.canvas.flush_events()


        epoch += 1

    if loadBestModel:
        #load best model; pathTrainedModels is defined in config
        model = torch.load(PurePath(pathTrainedModels, modelName +".pt")).to("cuda")
    else:
        torch.save(model, PurePath(pathTrainedModels, modelName +".pt"))

    return model, listU