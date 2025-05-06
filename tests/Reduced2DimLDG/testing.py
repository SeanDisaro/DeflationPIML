import matplotlib.pyplot as plt
from config import *
import torch
from pathlib import PurePath

def plotResults(model, branchFeatures, grid_N, showPlot = False):
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
    for i in range(n):
        ax[i%2, i//2].set_title(f'Plot {i}')
        ax[i%2, i//2].quiver(
            X, Y,                             # arrow tails
            U[i].detach().cpu().numpy(),
            V[i].detach().cpu().numpy(),      # arrow directions
            angles='xy', scale_units='xy', scale=20, width=0.0015
            )
    path = PurePath(plotFolder, "Reduced2DimLDG_Results.png")
    fig.savefig(path)
    if showPlot:
        fig.show()