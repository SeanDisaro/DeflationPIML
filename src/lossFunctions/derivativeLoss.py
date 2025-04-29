import torch

def derivativeLoss2D(x: torch.Tensor ,out1: list[torch.Tensor], out2: list[torch.Tensor], out1_dx: list[torch.Tensor], out2_dx: list[torch.Tensor], out1_dy: list[torch.Tensor], out2_dy: list[torch.Tensor] )->torch.Tensor:
    """Computes the derivative loss using auto differentiation.

    Args:
        x (torch.Tensor): Points wrt which we want to take the derivatives. Shape is (batchsize, 2)
        out1 (torch.Tensor): List of first component output tensors. The length of the list is the number of functions used as input in the DefDifONet model.
        out2 (torch.Tensor): List of first component output tensors. The length of the list is the number of functions used as input in the DefDifONet model.
        out1_dx (torch.Tensor): Partial derivative of first component in x direction computed by the model. 
        out2_dx (torch.Tensor): Partial derivative of second component in x direction computed by the model. 
        out1_dy (torch.Tensor): Partial derivative of first component in y direction computed by the model. 
        out2_dy (torch.Tensor): Partial derivative of second component in y direction computed by the model. 
        
    Returns:
        torch.Tensor: derivative loss.
    """
    batchSize = x.shape[0]
    batchSizeTimesNumInputfunc = batchSize *len(out1)

    out1_Concat = torch.concatenate(out1)
    out1_dxAD = torch.autograd.grad(out1_Concat.view(-1,1), x[:,0].view(-1,1),
                                     torch.ones((batchSizeTimesNumInputfunc, 1), requires_grad = True).to("cuda"),
                                     allow_unused=True, create_graph=True)[0]
    
    out2_Concat = torch.concatenate(out2)
    out2_dxAD = torch.autograd.grad(out2_Concat.view(-1,1), x[:,0].view(-1,1),
                                     torch.ones((batchSizeTimesNumInputfunc, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
    
    out1_dyAD = torch.autograd.grad(out1_Concat.view(-1,1), x[:,1].view(-1,1),
                                     torch.ones((batchSizeTimesNumInputfunc, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
    
    out2_dyAD = torch.autograd.grad(out2_Concat.view(-1,1), x[:,1].view(-1,1),
                                     torch.ones((batchSizeTimesNumInputfunc, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
    

    return (torch.mean(torch.norm(out1_dxAD - out1_dx,dim=1)) +
            torch.mean(torch.norm(out2_dxAD - out2_dx,dim=1)) +
            torch.mean(torch.norm(out1_dyAD - out1_dy,dim=1)) +
            torch.mean(torch.norm(out2_dyAD - out2_dy,dim=1)))