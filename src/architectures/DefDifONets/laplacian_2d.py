from torch import nn
import deepxde as dde
from typing import Callable, Tuple
import torch

class laplacian_2D_DefDifONet(nn.Module):
    """
    This model is a Deflation Diffusion DeepONet; The dimension of the out put
    is 2 and it outputs on top all the derivatives of the two outputs up to order 2 which we need for the laplacian.
    Furthermore, we output the branch features, since we want to reuse those for autoregression.
    Thus the total number of outputs of this model is 6.
    """
    def __init__(   self,
                    branch_layer: int,
                    branch_width: int,
                    numBranchFeatures: int,
                    trunk_layer: int,
                    trunk_width: int,
                    activationFunction: Callable[[torch.Tensor], torch.Tensor],
                    geom: dde.geometry.geometry.Geometry,
                    skipConnection: bool,
                    DirichletHardConstraint: bool,
                    sensorPoints: torch.Tensor,
                    DirichletConditionFunc: Callable[[torch.Tensor], torch.Tensor] = None
                    ):
        super().__init__()
        self.activationFunction = activationFunction
        self.numBranchFeatures = numBranchFeatures
        self.geom = geom
        self.DirichletHardConstraint = DirichletHardConstraint
        self.skipConnection = skipConnection
        self.sensorPoints = sensorPoints
        self.branch_layer = branch_layer
        self.DirichletConditionFunc = DirichletConditionFunc

        self.trunkNet = nn.Sequential(*
            (
                [nn.Linear(2,trunk_width), self.activationFunction] # input layer
                +
                [nn.Sequential(*[nn.Linear(trunk_width, trunk_width), self.activationFunction]) for i in range(max(trunk_layer-1,0))] # hidden layers
                +
                [nn.Linear(trunk_width, numBranchFeatures*6)] # output layers
            )
        )

        self.branch_Lin_OtherSol =  (     
                                      [nn.Linear(2, branch_width)]
                                    + [nn.Linear(branch_width, branch_width) for i in range( max(branch_layer-1,0) )]
                                    + [nn.Linear(branch_width, numBranchFeatures*6)]
                                    )

        self.branch_Lin          =  (     
                                      [nn.Linear(2, branch_width)]
                                    + [nn.Linear(branch_width, branch_width) for i in range( max(branch_layer-1,0) )]
                                    + [nn.Linear(branch_width, numBranchFeatures*6)]
                                    )

        self.branch_biases       =  (
                                      [nn.Parameter(torch.randn(branch_width))  for i in range( branch_layer )]
                                    + [nn.Parameter(torch.randn(numBranchFeatures*6))]
                                    )
        
        self.deepONet_biases     =    [nn.Parameter(torch.randn(1)) for i in range( 6 )]
                                
        
    def totalBranch(self, listU: list[torch.Tensor])->list[torch.Tensor]:
        n = len(listU)

        #initialize output
        outList = [listU[i].copy() for i in range(n)]

        for i in range(self.branch_layer):
            sumOthersol = torch.zeros_like(outList[0])
            for j in range(n):
                sumOthersol = sumOthersol + self.branch_Lin_OtherSol[i](outList[j])

            for j in range(n):
                outList[j] = sumOthersol - self.branch_Lin_OtherSol[i](outList[j])
                outList[j] = outList[j] + self.branch_Lin[i](outList[j])
                outList[j] = outList[j] + self.branch_biases[i]
                outList[j] = self.activationFunction( outList[j] )
        sumOthersol = torch.zeros_like(outList[0])
        for j in range(n):
            sumOthersol = sumOthersol + self.branch_Lin_OtherSol[-1](outList[j])

        for j in range(n):
            outList[j] = sumOthersol - self.branch_Lin_OtherSol[-1](outList[j])
            outList[j] = outList[j] + self.branch_Lin[-1](outList[j])
            outList[j] = outList[j] + self.branch_biases[-1]

        return outList

    #def trunk(self, x: torch.Tensor)->torch.Tensor:
    #    return self.trunkNet(x)

    def evaluateFunc(self, branchFeatures: torch.Tensor ,x: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        trunkOut = self.trunkNet(x)
        batchSize = trunkOut.shape[0]
        tiledBranchAux = torch.tile(branchFeatures, (batchSize,1))
        totalOutProd = trunkOut[:,:2*self.numBranchFeatures]*tiledBranchAux[:,:2*self.numBranchFeatures]
        out1 = (torch.sum(totalOutProd[:,:self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1)
        out2 = (torch.sum(totalOutProd[:,self.numBranchFeatures:], dim = 1) + self.deepONet_biases[1]).view(-1,1)
        return out1, out2


    def forward(self, listU: list[torch.Tensor], x: torch.Tensor)->Tuple[list[torch.Tensor],list[torch.Tensor],
                                                                         list[torch.Tensor],list[torch.Tensor],
                                                                         list[torch.Tensor],list[torch.Tensor],list[torch.Tensor]]:
        #output TrunkNet
        trunkOut = self.trunkNet(x)

        #output BranchNet
        n = len(listU)
        branchOut = self.totalBranch(listU)  
        batchSize = trunkOut.shape[0]
        branchTiledFeatures = []

        out1, out2, out1_dx, out2_dx, out1_dxx, out2_dxx = [],[],[],[],[],[]
        for i in range(n):
            tiledBranchAux = torch.tile(branchOut[i], (batchSize,1))
            branchTiledFeatures.append(tiledBranchAux )
            totalOutAux = trunkOut*tiledBranchAux 

            out1.       append( (torch.sum(totalOutAux[:,                         :   self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1) )
            out2.       append( (torch.sum(totalOutAux[:,   self.numBranchFeatures: 2*self.numBranchFeatures], dim = 1) + self.deepONet_biases[1]).view(-1,1) )
            out1_dx.    append( (torch.sum(totalOutAux[:, 2*self.numBranchFeatures: 3*self.numBranchFeatures], dim = 1) + self.deepONet_biases[2]).view(-1,1) )
            out2_dx.    append( (torch.sum(totalOutAux[:, 3*self.numBranchFeatures: 4*self.numBranchFeatures], dim = 1) + self.deepONet_biases[3]).view(-1,1) )
            out1_dxx.   append( (torch.sum(totalOutAux[:, 4*self.numBranchFeatures: 5*self.numBranchFeatures], dim = 1) + self.deepONet_biases[4]).view(-1,1) )
            out2_dxx.   append( (torch.sum(totalOutAux[:, 5*self.numBranchFeatures:                         ], dim = 1) + self.deepONet_biases[5]).view(-1,1) )



        return out1, out2, out1_dx, out2_dx, out1_dxx, out2_dxx, branchOut

