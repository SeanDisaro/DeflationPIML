import matplotlib.pyplot as plt
from config import *
import torch
from pathlib import PurePath

pathSavePictures = PurePath(plotFolder, "Reduced2DimLDG_ApproxDerivativeWithNN")


def plot_Q11_Q12(model, branchFeatures, grid_N, showPlot = False, saveName:str = "Reduced2DimLDG_Results" ):
    xs = torch.linspace(0.0, 1.0, grid_N, device="cpu")
    ys = torch.linspace(0.0, 1.0, grid_N, device="cpu")

    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    jointInputVec = torch.cat((X.reshape(-1,1),Y.reshape(-1,1)) , dim = 1).to("cuda")

    modelOut = model(branchFeatures, jointInputVec)

    U = modelOut[0]
    V = modelOut[1]


    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    n = len(U)
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].quiver(
                X, Y,                             # arrow tails
                U[i].detach().cpu().numpy(),
                V[i].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015*10/grid_N
                )
        

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].quiver(
                X, Y,                             # arrow tails
                U[i].detach().cpu().numpy(),
                V[i].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015*10/grid_N
                )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()




def plot_nematic_director(model, branchFeatures, grid_N, showPlot = False, saveName:str = "Reduced2DimLDG_nematic_director" ):
    xs = torch.linspace(0.0, 1.0, grid_N, device="cpu")
    ys = torch.linspace(0.0, 1.0, grid_N, device="cpu")

    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    jointInputVec = torch.cat((X.reshape(-1,1),Y.reshape(-1,1)) , dim = 1).to("cuda")

    modelOut = model(branchFeatures, jointInputVec)

    Q11 = modelOut[0]
    Q12 = modelOut[1]
    n = len(Q11)

    theta = []
    nematic_director = []
    for i in range(n):
        thetaAux = torch.arctan( Q12[i]/Q11[i] + (Q11[i] == 0).float() ) * (Q11[i] != 0).float()
        theta.append(thetaAux)
        nematic_director1 = torch.cos(thetaAux)
        nematic_director2 = torch.sin(thetaAux)
        nematic_director.append( torch.cat((nematic_director1,nematic_director2) , dim = 1) )

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].quiver(
                X, Y,                             # arrow tails
                nematic_director[i][:, 0].detach().cpu().numpy(),
                nematic_director[i][:, 1].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015*10/grid_N
                )
        

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].quiver(
                X, Y,                             # arrow tails
                nematic_director[i][:, 0].detach().cpu().numpy(),
                nematic_director[i][:, 1].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015*10/grid_N
                )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()