import torch
from src.lossFunctions.derivativeLoss import derivativeLoss2D, derivativeLoss
import logging
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
geomBoundaryExtension = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])

logger = logger = logging.getLogger("logging_config")

def pimlLoss_w_AutoGrad(modelOut:dict[list[torch.Tensor]],x, boundaryPoints:torch.Tensor = None, modelOutBoundary:dict[list[torch.Tensor]] = None ,
                eps:float = 0.02, alpha:float = 1., beta:float = 0.1):
    lossPDE = torch.tensor(0.)
    out1 = modelOut["out1"]
    out2 = modelOut["out2"]
    batchSize = x.shape[0]
    for i in range(len(out1)):
        out1_AD = torch.autograd.grad(out1[i].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
        
        out2_AD = torch.autograd.grad(out2[i].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
        
        out1_AD_dxx = torch.autograd.grad(out1_AD[:, 0].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out1_AD2_dyy = torch.autograd.grad(out1_AD[:, 1].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,1]
        
        out2_AD_dxx = torch.autograd.grad(out2_AD[:, 0].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out2_AD2_dyy = torch.autograd.grad(out2_AD[:, 1].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,1]


        laplace_comp1 = out1_AD_dxx + out1_AD2_dyy
        laplace_comp2 = out2_AD_dxx + out2_AD2_dyy
        Q_squaredNorm = out1[i] **2 + out2[i] **2
        lossPDE_comp1 = ((2*   (1 - Q_squaredNorm) *out1[i]).view(-1)    + (eps **2)*laplace_comp1)
        lossPDE_comp2 = ((2*   (1 - Q_squaredNorm) *out2[i]).view(-1)    + (eps **2)*laplace_comp2)
        lossPDE = lossPDE +  torch.nanmean(torch.abs(lossPDE_comp1)) + torch.nanmean(torch.abs(lossPDE_comp2 ))

    if boundaryPoints != None and modelOutBoundary != None:
        out1B = modelOutBoundary["out1"]
        out2B = modelOutBoundary["out2"]
        outTrue1 = boundaryFunctionExtension(boundaryPoints, 3*eps)
        outTrue2 = torch.zeros((boundaryPoints.shape[0], 1))
        boundaryLoss = torch.tensor(0.)
        for i in range(len(out1B)):
            boundaryLoss = boundaryLoss +  torch.nanmean(torch.norm(out1B[i] - outTrue1 , dim = 1)) + torch.nanmean(torch.norm(out2B[i] - outTrue2, dim = 1))
        return  alpha* lossPDE + beta * boundaryLoss
    else:
        return alpha* lossPDE



def defDifONetLossPIML_w_AD( x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None,modelOutBoundary:list[list[torch.Tensor]] = None ,
                    eps:float = 0.02, deflationLossPoints: tuple[float,float] = (10000.,1.) ,deflationLossCoeff:float = 1., alpha:float = 1., beta:float = 0.1, delta: float = 1.) -> torch.Tensor:

    loss_PDEAndBoundary = pimlLoss_w_AutoGrad(  modelOut = modelOut,x=x, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                                    eps = eps, alpha = alpha, beta = beta)

    deflationLossOut = linearDeflationLoss_dictModel(modelOut = modelOut, maxLoss = deflationLossPoints[0], maxDistance = deflationLossPoints[1]) #deflationLoss(modelOut=modelOut, a = deflationLossCoeff)
    logger.info("--------------------------------------------------------------------------------------------------------------------------------")
    logger.info(f"PDE Loss: {loss_PDEAndBoundary.item():.2f}--------Deflation Loss: {delta*deflationLossOut.item():.2f}")
    return loss_PDEAndBoundary + delta*deflationLossOut

def pimlLoss(modelOut:list[list[torch.Tensor]],x, boundaryPoints:torch.Tensor = None, modelOutBoundary:list[list[torch.Tensor]] = None ,
                eps:float = 0.02, alpha:float = 1., beta:float = 0.1)->torch.Tensor:
    """This is the physics informed loss function for this problem, which is created with the strong formulation of the PDE.

    Args:
        modelOut (list[list[torch.Tensor]]): This is the output of the model.
        boundaryPoints (boundaryPoints, optional): Set of boundary points. This is only used if boundaryLoss is set to True. Defaults to None. If this is set to None, then only the boundary loss term is omitted
        modelOutBoundary (list[list[torch.Tensor]], optional): Result of model on the boundary. If this is None, then the boundary loss term is omitted.
        eps (float, optional): epsilon parameter form paper. Defaults to 0.02.
        alpha (float, optional): Weight parameter loss term PDE.
        beta (float, optional):Weight parameter loss term boundary.

    Returns:
        torch.Tensor: loss
    """
    lossPDE = torch.tensor(0.)
    for i in range(len(modelOut[0])):
        dummy = False
        laplace_comp1 = modelOut[4][i] + modelOut[8][i]
        laplace_comp2 = modelOut[5][i] + modelOut[9][i]
        Q_squaredNorm = modelOut[0][i] **2 + modelOut[1][i] **2
        lossPDE_comp1 = 200*   (1 - Q_squaredNorm) *modelOut[0][i]    + ((10*eps) **2)*laplace_comp1
        lossPDE_comp2 = 200*   (1 - Q_squaredNorm) *modelOut[1][i]    + ((10*eps) **2)*laplace_comp2
        lossPDE = lossPDE +  torch.nanmean(torch.norm(lossPDE_comp1, dim = 1)) + torch.nanmean(torch.norm(lossPDE_comp2, dim = 1)) 
        
        if dummy:
            plt.clf()
            plt.quiver(
                x[:,0].detach().cpu().numpy()[:200], x[:,1].detach().cpu().numpy()[:200],
                lossPDE_comp1.detach().cpu().numpy()[:200],
                lossPDE_comp2.detach().cpu().numpy()[:200],      
                angles='xy', scale_units='xy', scale=20, width=0.0015*10/33
                )
            plt.show()

    if boundaryPoints != None and modelOutBoundary != None:
        out1B = modelOutBoundary[0]
        out2B = modelOutBoundary[1]
        outTrue1 = boundaryFunctionExtension(boundaryPoints, 3*eps)
        outTrue2 = torch.zeros((boundaryPoints.shape[0], 1))
        boundaryLoss = torch.tensor(0.)
        for i in range(len(out1B)):
            boundaryLoss = boundaryLoss +  torch.nanmean(torch.norm(out1B[i] - outTrue1 , dim = 1)) + torch.nanmean(torch.norm(out2B[i] - outTrue2, dim = 1))
        return  alpha* lossPDE + beta * boundaryLoss
    else:
        return alpha* lossPDE
 


def defDifONetLossPIML( x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None,modelOutBoundary:list[list[torch.Tensor]] = None ,
                    eps:float = 0.02, deflationLossPoints: tuple[float,float] = (10000.,1.) ,deflationLossCoeff:float = 1., alpha:float = 1., beta:float = 0.1, gamma:float = 1., delta: float = 1.) -> torch.Tensor:
    """This is the loss funciton you want to use for the DefDifONet. This includes losses for the reduced 2dim LDG PDE, derivative losses and a boundary loss (optional). Derivative loss is computed via auto differentiation.
    Args:
        x (torch.Tensor): points which were put into modelOut. Note that requires_grad needs to be set to true.
        modelOut (list[list[torch.Tensor]]): Output of the model.
        boundaryPoints (torch.Tensor, optional): Boundary Points that were put into the model for computing output of model on the boundary. If this is set to None, then no boundary loss is computed. Defaults to None.
        modelOutBoundary (list[torch.Tensor], optional): Output of model on the boundary. If this is set to None, then no boundary loss is computed. Defaults to None.
        eps (float, optional): epsilon from LDG model, as in paper. Defaults to 0.02.
        eps (float, optional): Deflation loss coefficient, see deflation loss. Defaults to 1.
        deflationLossPoints (tuple[float,float], optional): Points adjusting the linear deflation loss function. This contains (maxLoss, maxDistance). Defaults to (10000.,1.).
        deflationLossCoeff (float, optional): Deflation coefficient from original deflation loss function. Defaults to 1.
        alpha (float, optional): Weigth of PDE loss in total loss funciton. Defaults to 1..
        beta (float, optional): Weight of boundary loss in total loss function. Defaults to 0.1.
        gamma (float, optional): Weight of derivative loss in total loss funciton. Defaults to 1.
        delta (float, optional): Weight of deflation loss in total loss funciton. Defaults to 1.
    Returns:
        torch.Tensor: alpha*PDE_loss + beta* boundary_loss+ gamma*(1stOrderDerivatives_loss+ 2ndOrderDerivative_loss) + delta* deflationLoss
    """
    loss_PDEAndBoundary = pimlLoss(  modelOut = modelOut,x=x, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                                    eps = eps, alpha = alpha, beta = beta)

    # derivative of first component makes problems because of hard constraint architecture, so we have to treat its derivatives differently.
    dx_Order_Comp1_pre_transform  =   derivativeLoss(x = x,out = modelOut[0], out_dx = modelOut[2], component = 0, p = 1 ) 
    dy_Order_Comp1_pre_transform  =   derivativeLoss(x = x,out = modelOut[0], out_dx = modelOut[6], component = 1, p = 1 ) 
    #logger.info(f"dx_Order_Comp1:.......{dx_Order_Comp1_pre_transform:.2f}...........................dy_Order_Comp1:.......{dy_Order_Comp1_pre_transform:.2f}")
    dx_Order_Comp1  =  dx_Order_Comp1_pre_transform 
    dy_Order_Comp1  = dy_Order_Comp1_pre_transform 
    #logger.info(f"dx_Order_Comp1_Transformed:.......{dx_Order_Comp1:.2f}...........................dy_Order_Comp1_Transformed:.......{dy_Order_Comp1:.2f}")


    dx_Order_Comp2  = derivativeLoss(x = x,out = modelOut[1], out_dx = modelOut[3], component = 0 )
    dy_Order_Comp2  = derivativeLoss(x = x,out = modelOut[1], out_dx = modelOut[7], component = 1 )

    dxx_Order_Comp1 = derivativeLoss(x = x,out = modelOut[2], out_dx = modelOut[4], component = 0 )
    dxx_Order_Comp2 = derivativeLoss(x = x,out = modelOut[3], out_dx = modelOut[5], component = 0 )
    dyy_Order_Comp1 = derivativeLoss(x = x,out = modelOut[6], out_dx = modelOut[8], component = 1 )
    dyy_Order_Comp2 = derivativeLoss(x = x,out = modelOut[7], out_dx = modelOut[9], component = 1 )

    totalLossDerivatives = (dx_Order_Comp1 + dy_Order_Comp1 + dx_Order_Comp2 + dy_Order_Comp2 + dxx_Order_Comp1 + dxx_Order_Comp2 + dyy_Order_Comp1 + dyy_Order_Comp2)



    deflationLossOut = linearDeflationLoss(modelOut = modelOut, maxLoss = deflationLossPoints[0], maxDistance = deflationLossPoints[1]) #deflationLoss(modelOut=modelOut, a = deflationLossCoeff)
    logger.info("--------------------------------------------------------------------------------------------------------------------------------")
    logger.info(f"PDE Loss: {loss_PDEAndBoundary.item():.2f}----Derivative Loss: {gamma*totalLossDerivatives.item():.2f}----Deflation Loss: {delta*deflationLossOut.item():.2f}")
    totalLoss = loss_PDEAndBoundary + gamma*totalLossDerivatives + delta* deflationLossOut
    return totalLoss


def defDifONetLossPIML_ModelWithOutDerivative(  x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None,modelOutBoundary:list[list[torch.Tensor]] = None ,
                                                eps:float = 0.02, deflationLossPoints: tuple[float,float] = (10000.,1.) ,deflationLossCoeff:float = 1., alpha:float = 1., beta:float = 0.1,
                                                  gamma:float = 1., delta: float = 1.) -> torch.Tensor:
    pass


def trapezoidalFun(x: torch.Tensor, d:float)->torch.Tensor:
    """This is the funciton T_d(x) from the paper.

    Args:
        x (torch.Tensor): one dimensional x values.
        d (float): parameter for the function.

    Returns:
        torch.Tensor: T_d(x) as in paper.
    """
    out = torch.zeros_like(x,dtype=float)
    maskInterval_0_to_d       = (x >=0)   * (x<=d)
    maskInterval_d_to_1Minusd = (x >d)      * (x<(1-d))
    maskInterval_1Minusd_to_1 = (x >=(1-d)) * (x<=1)
    out = maskInterval_0_to_d       * (x / d)
    out = out + maskInterval_d_to_1Minusd * 1
    out = out + maskInterval_1Minusd_to_1 * ((1-x)/d)
    return out

def boundaryFunctionExtension(x: torch.Tensor, d:float)->torch.Tensor:
    """This is an extension of the first component of the boudndary funciton Q_b form the paper.
        The extension is of the form T_d(x) * x(1-x)/ (x(1-x) + y(1-y)) - T_d(y) * y(1-y)/ (x(1-x) + y(1-y)).

    Args:
        x (torch.Tensor): Two dimensional input data vecotrs.
        d (float): parameter for the funciton.

    Returns:
        torch.Tensor: funciton value of extension.
    """    
    x1 = x[:, 0]
    x2 = x[:, 1]
    #this mask stuff is to make sure, that we do not devide by 0.
    maskZeroDenominator = (x1* (1-x1) == 0) * (x2* (1-x2) == 0)
    maskZeroDenominatorNegation = maskZeroDenominator == False
    out = trapezoidalFun(x1, d)* (x1* (1-x1)) / ((x1* (1-x1)) + (x2* (1-x2)) + maskZeroDenominator) - trapezoidalFun(x2, d)* (x2* (1-x2)) / ((x1* (1-x1)) + (x2* (1-x2)) + maskZeroDenominator)
    out = out * maskZeroDenominatorNegation
    # phi1 = (x1 **2 *(1- x1)**2).view(-1,1)   
    # phi2 = (x2 **2 *(1- x2)**2).view(-1,1)
    # alpha = phi1 / (phi1 + phi2)
    out = trapezoidalFun(x1, d).view(-1,1)   -  trapezoidalFun(x2, d).view(-1,1)
    return out


def trapezoidalFunDerivative(x: torch.Tensor, d:float)->torch.Tensor:
    """This is the derivative T'_d(x) for the function T_d(x) from the paper.

    Args:
        x (torch.Tensor): one dimensional x values.
        d (float): parameter for the function.

    Returns:
        torch.Tensor: T'_d(x).
    """
    oneOverD = 1 / d
    out = torch.zeros_like(x,dtype=float)
    maskInterval_0_to_d       = (x >=0)   * (x<=d)
    maskInterval_1Minusd_to_1 = (x >=(1-d)) * (x<=1)
    out = maskInterval_0_to_d  * oneOverD
    out = out + maskInterval_1Minusd_to_1 *(- oneOverD)
    return out




def deflationLoss(modelOut:list[list[torch.Tensor]], a:float=1)->torch.Tensor:
    """Deflation loss function to make sure that output functions are different from each other.
        we basically compute sum_{i,j}( 1/(a* norm(func_i - func_j))^a.

    Args:
        modelOut (list[list[torch.Tensor]]): Output of our model.
        a (float, optional): Coefficient for deflation funciton. If a gets bigger, then the L2 distance for the function is allowed to be bigger and we still get a low loss. Defaults to 1.

    Returns:
        torch.Tensor: Deflation loss.
    """ 
    n = len(modelOut[0])
    loss = torch.Tensor([0.]).to(modelOut[0][0].device)
    if n == 1:
        return loss
    for i in range(n):
        for j in range(n-1-i):
            difference_ij_0 = modelOut[0][i] - modelOut[0][i + j+1]
            difference_ij_1 = modelOut[1][i] - modelOut[1][i + j+1]
            lossAux1 = torch.min(torch.maximum(10000-torch.pow(a* torch.nanmean(torch.norm(difference_ij_0 , dim = 1)), a ), torch.tensor(0.)),1/ torch.pow(a* torch.nanmean(torch.norm(difference_ij_0 , dim = 1)), a ))
            lossAux2 = torch.min(torch.maximum(10000-torch.pow(a* torch.nanmean(torch.norm(difference_ij_1 , dim = 1)), a ), torch.tensor(0.)), 1/ torch.pow(a* torch.nanmean(torch.norm(difference_ij_1 , dim = 1)), a ))
            loss = loss + lossAux2 + lossAux1
    
    loss = 2*loss /(n* (n-1)) 
    return loss





def linearDeflationLoss(modelOut:list[list[torch.Tensor]], maxLoss:float=10000, maxDistance:float = 0.1)->torch.Tensor:
    """Linear deflation function. We compute basically sum _i,j max(maxLoss - maxLoss/ maxDistance * dist(u_i, u_j), 0  )

    Args:
        modelOut (list[list[torch.Tensor]]): Output of our model.
        maxLoss (float, optional): Maximal possible loss. Defaults to 10000.
        maxDistance (float, optional): After dist(u_i, u_j) gets to this value, the loss becomes 0 and we stop optimizing for deflation. Defaults to 0.1.

    Returns:
        torch.Tensor: _description_
    """    
    n = len(modelOut[0])
    m = maxLoss/maxDistance
    loss = torch.Tensor([0.]).to(modelOut[0][0].device)
    if n == 1:
        return loss
    for i in range(n):
        for j in range(n-1-i):
            difference_ij_0 = modelOut[0][i] - modelOut[0][i + j+1]
            difference_ij_1 = modelOut[1][i] - modelOut[1][i + j+1]
            lossAux1 = torch.maximum(maxLoss - m* torch.nanmean(torch.norm(difference_ij_0 , dim = 1)), torch.tensor(0.))
            lossAux2 = torch.maximum(maxLoss - m* torch.nanmean(torch.norm(difference_ij_1 , dim = 1)), torch.tensor(0.))
            loss = loss + lossAux2 + lossAux1
    
    loss = 2*loss /(n* (n-1)) 
    return loss


def linearDeflationLoss_dictModel(modelOut:dict[list[torch.Tensor]], maxLoss:float=10000, maxDistance:float = 0.1)->torch.Tensor:
    """Linear deflation function. We compute basically sum _i,j max(maxLoss - maxLoss/ maxDistance * dist(u_i, u_j), 0  )

    Args:
        modelOut (list[list[torch.Tensor]]): Output of our model.
        maxLoss (float, optional): Maximal possible loss. Defaults to 10000.
        maxDistance (float, optional): After dist(u_i, u_j) gets to this value, the loss becomes 0 and we stop optimizing for deflation. Defaults to 0.1.

    Returns:
        torch.Tensor: _description_
    """
    #out1 = modelOut["out1"]
    out2 = modelOut["out2"]
    n = len(out2)
    m = maxLoss/maxDistance
    loss = torch.Tensor([0.]).to(out2[0].device)
    if n == 1:
        return loss
    for i in range(n):
        for j in range(n-1-i):
            #difference_ij_0 = out1[i] - out1[i + j+1]
            difference_ij_1 = out2[i] - out2[i + j+1]
            #lossAux1 = torch.maximum(maxLoss - m* torch.mean(torch.norm(difference_ij_0 , dim = 1)), torch.tensor(0.))
            lossAux2 = torch.maximum((maxLoss - m* torch.nanmean(torch.norm(difference_ij_1 , dim = 1))) , torch.tensor(0.))
            loss = loss + lossAux2 #+ lossAux1
    
    loss = 2*loss /(n* (n-1))
    return loss