from torch import nn
import deepxde as dde
from typing import Callable, Tuple
import torch

class Laplacian_2D_DefDifONet(nn.Module):
    """
    This model is a Deflation Diffusion DeepONet; The dimension of the out put
    is 2 and it outputs on top all the derivatives of the two outputs up to order 2 which we need for the laplacians of the components.
    Furthermore, we output the branch features, since we want to reuse those for autoregression.
    Thus the total number of outputs of this model is 11.
    """
    def __init__(   self,
                    branch_layer: int,
                    branch_width: int,
                    numBranchFeatures: int,
                    trunk_layer: int,
                    trunk_width: int,
                    activationFunction: Callable[[torch.Tensor], torch.Tensor],
                    geom: dde.geometry.geometry.Geometry,
                    DirichletHardConstraint: bool,
                    skipConnection: bool = False,
                    normalizationBranch: bool = False,
                    normalizationFactor: float = 10.,
                    DirichletConditionFunc1: Callable[[torch.Tensor], torch.Tensor] = None,
                    DirichletConditionFunc2: Callable[[torch.Tensor], torch.Tensor] = None
                    ):
        super().__init__()
        self.activationFunction = activationFunction
        self.numBranchFeatures = numBranchFeatures
        self.geom = geom
        self.DirichletHardConstraint = DirichletHardConstraint
        self.skipConnection = skipConnection
        self.branch_layer = branch_layer
        self.DirichletConditionFunc1 = DirichletConditionFunc1
        self.DirichletConditionFunc2 = DirichletConditionFunc2
        self.trunk_layer = trunk_layer
        self.normalizationBranch = normalizationBranch
        self.normalizationFactor = normalizationFactor



        self.trunkNet_Lin        =  (     
                                      [nn.Linear(2,trunk_width)]
                                    + [nn.Linear(trunk_width, trunk_width) for i in range(max(trunk_layer-1,0))]
                                    + [nn.Linear(trunk_width, numBranchFeatures*10)]
                                    )
        
        # add the modules manually to ensure, that all parameters appear in model paramters
        for idx,module in enumerate(self.trunkNet_Lin):
            self.add_module(f"trunkNet_Lin_{idx}", module)


        self.branch_Lin_OtherSol =  (     
                                      [nn.Linear(numBranchFeatures*10, branch_width, bias=False)]
                                    + [nn.Linear(branch_width, branch_width, bias=False) for i in range( max(branch_layer-1,0) )]
                                    + [nn.Linear(branch_width, numBranchFeatures*10, bias=False)]
                                    )
        
        # add the modules manually to ensure, that all parameters appear in model paramters
        for idx,module in enumerate(self.branch_Lin_OtherSol):
            self.add_module(f"branch_Lin_OtherSol_{idx}", module)

        self.branch_Lin          =  (     
                                      [nn.Linear(numBranchFeatures*10, branch_width)]
                                    + [nn.Linear(branch_width, branch_width) for i in range( max(branch_layer-1,0) )]
                                    + [nn.Linear(branch_width, numBranchFeatures*10)]
                                    )

        # add the modules manually to ensure, that all parameters appear in model paramters
        for idx,module in enumerate(self.branch_Lin):
            self.add_module(f"branch_Lin_{idx}", module)

        
        self.deepONet_biases     =    nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range( 10 )])


        
    def totalBranch(self, listU: list[torch.Tensor])->list[torch.Tensor]:
        n = len(listU)

        #initialize output
        outList = [listU[i].clone() for i in range(n)]

        for i in range(self.branch_layer):
            sumOthersol = torch.zeros((self.branch_Lin_OtherSol[i].out_features), device=outList[0].device)
            for j in range(n):
                sumOthersol = sumOthersol + self.branch_Lin_OtherSol[i](outList[j])

            for j in range(n):
                skipConn = outList[j].clone()
                otherSolContribution = sumOthersol - self.branch_Lin_OtherSol[i](outList[j])
                outList[j] = otherSolContribution + self.branch_Lin[i](outList[j])
                outList[j] = self.activationFunction( outList[j] )
                if self.skipConnection and i != 0:
                    outList[j] = skipConn + outList[j]
        sumOthersol = torch.zeros((self.branch_Lin_OtherSol[-1].out_features), device=outList[0].device )
        for j in range(n):
            sumOthersol = sumOthersol + self.branch_Lin_OtherSol[-1](outList[j])

        for j in range(n):
            otherSolContribution = sumOthersol - self.branch_Lin_OtherSol[-1](outList[j])
            outList[j] = otherSolContribution + self.branch_Lin[-1](outList[j])
        return outList

    def trunk(self, x: torch.Tensor)->torch.Tensor:
        out = torch.zeros_like(x, device= x.device) + x
        for i in range(self.trunk_layer):
            skipConn = out.clone()
            out = self.activationFunction(self.trunkNet_Lin[i](out))
            if self.skipConnection and i != 0:
                out = out + skipConn
        
        out = self.trunkNet_Lin[-1](out)

        return out

    def evaluateFunc(self, branchFeatures: torch.Tensor ,x: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        trunkOut = self.trunk(x)
        batchSize = trunkOut.shape[0]
        tiledBranchAux = torch.tile(branchFeatures, (batchSize,1))
        totalOutProd = trunkOut[:,:2*self.numBranchFeatures]*tiledBranchAux[:,:2*self.numBranchFeatures]
        out1 = (torch.sum(totalOutProd[:,:self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1)
        out2 = (torch.sum(totalOutProd[:,self.numBranchFeatures:], dim = 1) + self.deepONet_biases[1]).view(-1,1)
        if self.DirichletHardConstraint:
            out1 = out1 * self.geom.boundary_constraint_factor(x, smoothness="Cinf")
            out2 = out2 * self.geom.boundary_constraint_factor(x, smoothness="Cinf")
            if self.DirichletConditionFunc1 != None:
                out1 = out1 + self.DirichletConditionFunc1(x).view(-1,1)
            if self.DirichletConditionFunc2 != None:
                out2 = out2 + self.DirichletConditionFunc2(x).view(-1,1)
        return out1, out2


    def forward(self, listU: list[torch.Tensor], x: torch.Tensor)->Tuple[list[torch.Tensor],list[torch.Tensor],
                                                                         list[torch.Tensor],list[torch.Tensor],
                                                                         list[torch.Tensor],list[torch.Tensor],
                                                                         list[torch.Tensor],list[torch.Tensor],
                                                                         list[torch.Tensor],list[torch.Tensor],list[torch.Tensor]]:
        #xClone = x.clone()
        #output TrunkNet
        trunkOut = self.trunk(x)

        #output BranchNet
        n = len(listU)
        branchOut = self.totalBranch(listU)
        if self.normalizationBranch:
            for i in range(n):
                branchOut[i] = self.normalizationFactor * torch.nn.functional.softsign(branchOut[i])
        batchSize = trunkOut.shape[0]
        branchTiledFeatures = []

        out1, out2, out1_dx, out2_dx, out1_dxx, out2_dxx, out1_dy, out2_dy, out1_dyy, out2_dyy = [],[],[],[],[],[],[],[],[],[]

        for i in range(n):
            tiledBranchAux = torch.tile(branchOut[i], (batchSize,1))
            branchTiledFeatures.append(tiledBranchAux )
            totalOutAux = trunkOut*tiledBranchAux

            out1.       append( ( torch.sum(totalOutAux[:,                         :   self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1) )
            out2.       append( ( torch.sum(totalOutAux[:,   self.numBranchFeatures: 2*self.numBranchFeatures], dim = 1) + self.deepONet_biases[1]).view(-1,1) )
            out1_dx.    append( ( torch.sum(totalOutAux[:, 2*self.numBranchFeatures: 3*self.numBranchFeatures], dim = 1) + self.deepONet_biases[2]).view(-1,1)) 
            out2_dx.    append( ( torch.sum(totalOutAux[:, 3*self.numBranchFeatures: 4*self.numBranchFeatures], dim = 1) + self.deepONet_biases[3]).view(-1,1) )
            out1_dxx.   append( ( torch.sum(totalOutAux[:, 4*self.numBranchFeatures: 5*self.numBranchFeatures], dim = 1) + self.deepONet_biases[4]).view(-1,1) )
            out2_dxx.   append( ( torch.sum(totalOutAux[:, 5*self.numBranchFeatures: 6*self.numBranchFeatures], dim = 1) + self.deepONet_biases[5]).view(-1,1) )
            out1_dy.    append( ( torch.sum(totalOutAux[:, 6*self.numBranchFeatures: 7*self.numBranchFeatures], dim = 1) + self.deepONet_biases[6]).view(-1,1)) 
            out2_dy.    append( ( torch.sum(totalOutAux[:, 7*self.numBranchFeatures: 8*self.numBranchFeatures], dim = 1) + self.deepONet_biases[7]).view(-1,1))
            out1_dyy.   append( ( torch.sum(totalOutAux[:, 8*self.numBranchFeatures: 9*self.numBranchFeatures], dim = 1) + self.deepONet_biases[8]).view(-1,1) )
            out2_dyy.   append( ( torch.sum(totalOutAux[:, 9*self.numBranchFeatures:                         ], dim = 1) + self.deepONet_biases[9]).view(-1,1) )
        
        # these are used for training, when the hardconstraint addition term has a very compicated derivative for autodifferentiation to grasp.
        #out1_preHardConst = []
        #out2_preHardConst = []

        if self.DirichletHardConstraint:
            for idxSol in range(n):
                out1[idxSol] = out1[idxSol] * self.geom.boundary_constraint_factor(x, smoothness="Cinf")
                out2[idxSol] = out2[idxSol] * self.geom.boundary_constraint_factor(x, smoothness="Cinf")
                if self.DirichletConditionFunc1 != None:
                    #out1_preHardConst.append(out1[idxSol].clone())
                    out1[idxSol] = out1[idxSol] + self.DirichletConditionFunc1(x).view(-1,1)
                if self.DirichletConditionFunc2 != None:
                    #out2_preHardConst.append(out1[idxSol].clone())
                    out2[idxSol] = out2[idxSol] + self.DirichletConditionFunc2(x).view(-1,1)

        return out1, out2, out1_dx, out2_dx, out1_dxx, out2_dxx, out1_dy, out2_dy, out1_dyy, out2_dyy, branchOut#, out1_preHardConst, out2_preHardConst

