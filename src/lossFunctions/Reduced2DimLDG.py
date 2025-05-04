import torch
from src.lossFunctions.derivativeLoss import derivativeLoss2D, derivativeLoss

def pimlLoss(modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None, modelOutBoundary:list[list[torch.Tensor]] = None ,
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
    laplace_comp1 = torch.concatenate(modelOut[4]) + torch.concatenate(modelOut[8])
    laplace_comp2 = torch.concatenate(modelOut[5]) + torch.concatenate(modelOut[9])
    Q_squaredNorm = torch.concatenate(modelOut[0])**2 + torch.concatenate(modelOut[1])**2
    lossPDE_comp1 = 2*   (1 - Q_squaredNorm) *torch.concatenate(modelOut[0])    / (eps **2) + laplace_comp1
    lossPDE_comp2 = 2*   (1 - Q_squaredNorm) *torch.concatenate(modelOut[1])    / (eps **2) + laplace_comp2
    lossPDE = torch.mean(torch.norm(lossPDE_comp1, dim = 1)) + torch.mean(torch.norm(lossPDE_comp2, dim = 1)) 
    if boundaryPoints == None or modelOutBoundary == None:
        boundaryLoss = (    torch.mean(torch.norm( torch.concatenate(modelOutBoundary[0]) - trapezoidalFun(boundaryPoints, 3*eps), dim = 1))
                            + torch.mean(torch.norm( torch.concatenate(modelOutBoundary[1]) , dim = 1)))
        return alpha* lossPDE + beta  * boundaryLoss
    else:
        return lossPDE
    


def defDifONetLossPIML( x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None,modelOutBoundary:list[list[torch.Tensor]] = None ,
                    eps:float = 0.02, deflationLossCoeff:float = 1., alpha:float = 1., beta:float = 0.1, gamma:float = 1., delta: float = 1.) -> torch.Tensor:
    """This is the loss funciton you want to use for the DefDifONet. This includes losses for the reduced 2dim LDG PDE, derivative losses and a boundary loss (optional). Derivative loss is computed via auto differentiation.
    Args:
        x (torch.Tensor): points which were put into modelOut. Note that requires_grad needs to be set to true.
        modelOut (list[list[torch.Tensor]]): Output of the model.
        boundaryPoints (torch.Tensor, optional): Boundary Points that were put into the model for computing output of model on the boundary. If this is set to None, then no boundary loss is computed. Defaults to None.
        modelOutBoundary (list[torch.Tensor], optional): Output of model on the boundary. If this is set to None, then no boundary loss is computed. Defaults to None.
        eps (float, optional): epsilon from LDG model, as in paper. Defaults to 0.02.
        eps (float, optional): Deflation loss coefficient, see deflation loss. Defaults to 1.
        alpha (float, optional): Weigth of PDE loss in total loss funciton. Defaults to 1..
        beta (float, optional): Weight of boundary loss in total loss function. Defaults to 0.1.
        gamma (float, optional): Weight of derivative loss in total loss funciton. Defaults to 1.
        delta (float, optional): Weight of deflation loss in total loss funciton. Defaults to 1.
    Returns:
        torch.Tensor: alpha*PDE_loss + beta* boundary_loss+ gamma*(1stOrderDerivatives_loss+ 2ndOrderDerivative_loss) + delta* deflationLoss
    """
    loss_PDEAndBoundary = pimlLoss(  modelOut = modelOut, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                                    eps = eps, alpha = alpha, beta = beta)
    loss_FirstOrderDerivatives = derivativeLoss2D(x=x, out1 = modelOut[0], out2 = modelOut[1], out1_dx = modelOut[2], out2_dx = modelOut[3], out1_dy = modelOut[6], out2_dy = modelOut[7])/4

    dxx_Order_Comp1 = derivativeLoss(x = x,out = modelOut[2], out_dx = modelOut[4], component = 0 )
    dxx_Order_Comp2 = derivativeLoss(x = x,out = modelOut[3], out_dx = modelOut[5], component = 0 )
    dyy_Order_Comp1 = derivativeLoss(x = x,out = modelOut[6], out_dx = modelOut[8], component = 1 )
    dyy_Order_Comp2 = derivativeLoss(x = x,out = modelOut[7], out_dx = modelOut[9], component = 1 )

    secondOrderLossTotal = dxx_Order_Comp1 + dxx_Order_Comp2 + dyy_Order_Comp1 + dyy_Order_Comp2
    secondOrderLossTotal = secondOrderLossTotal/4

    totalLossDerivatives = ( loss_FirstOrderDerivatives + secondOrderLossTotal )/2

    totalLoss = loss_PDEAndBoundary + gamma*totalLossDerivatives + delta* deflationLoss(modelOut=modelOut, a = deflationLossCoeff)
    return totalLoss
    

def trapezoidalFun(x: torch.Tensor, d:float)->torch.Tensor:
    """This is the funciton T_d(x) from the paper.

    Args:
        x (torch.Tensor): one dimensional x values.
        d (float): parameter for the function.

    Returns:
        torch.Tensor: T_d(x) as in paper.
    """
    out = torch.zeros_like(x,dtype=float)
    maskInterval_0_to_d       = x >=0     * x<=d
    maskInterval_d_to_1Minusd = x >d      * x<(1-d)
    maskInterval_1Minusd_to_1 = x >=(1-d) * x<=1
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
    maskZeroDenominatorNegation = not maskZeroDenominatorNegation
    out = trapezoidalFun(x1, d)* (x1* (1-x1)) / ((x1* (1-x1)) + (x2* (1-x2)) + maskZeroDenominator) - trapezoidalFun(x2, d)* (x2* (1-x2)) / ((x1* (1-x1)) + (x2* (1-x2)) + maskZeroDenominator)
    out = out * maskZeroDenominatorNegation
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
    loss = torch.Tensor([0.], device=modelOut[0][0].device)
    for i in range(n):
        for j in range(n-1-i):
            difference_ij_0 = modelOut[0][i] - modelOut[0][i + j]
            difference_ij_1 = modelOut[1][i] - modelOut[1][i + j]
            lossAux1 = 1/ torch.pow(a* torch.mean(torch.norm(difference_ij_0 , dim = 1)), a )
            lossAux2 = 1/ torch.pow(a* torch.mean(torch.norm(difference_ij_1 , dim = 1)), a )
            loss = loss + lossAux2 + lossAux1
    
    loss = 2*loss /(n* (n-1)) 
    return loss