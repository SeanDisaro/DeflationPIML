import torch

def pimlLoss(x: torch.Tensor, model:torch.nn.Module, boundaryPoints:torch.Tensor = None ,boundaryLoss:bool = False , eps:float = 0.02)->torch.Tensor:
    """This is the physics informed loss function for this problem, which is created with the strong formulation of the PDE.

    Args:
        x (torch.Tensor): This is the set of points in the interior of the domain.
        model (torch.nn.Module): This is the model you want to train. The model should return two dimensional values and their
                                 first and second derivatives w.r.t their direction (i.e. only the directions needed for the laplacian).
        boundaryPoints (boundaryPoints, optional): Set of boundary points. This is only used if boundaryLoss is set to True. Defaults to None.
        boundaryLoss (bool, optional): If you want to learn the boundary condition via softconstraint set this to True. Defaults to False.
        eps (float, optional): epsilon parameter form paper. Defaults to 0.02.

    Returns:
        torch.Tensor: loss
    """    
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