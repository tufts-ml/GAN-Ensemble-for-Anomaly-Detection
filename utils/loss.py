import torch
from torch import nn
from torch.autograd import Variable

class con_loss(nn.Module):
    def __init__(self, b, reduction='mean'):
        super(con_loss, self).__init__()
        self.b = torch.tensor(b)
        if reduction in ['sum', 'mean']:
            self.reduction = reduction
        else:
            raise KeyError

    def forward(self, x, target):
        if self.reduction == 'mean':
            return torch.mean(torch.abs(x-target)) / self.b
        else:
            return torch.abs(x-target) / self.b


class lat_loss(nn.Module):
    def __init__(self, sigma, reduction='mean'):
        super(lat_loss, self).__init__()
        self.sigma = torch.tensor(sigma)
        if reduction in ['sum', 'mean']:
            self.reduction = reduction
        else:
            raise KeyError

    def forward(self, x, target):
        if self.reduction == 'mean':
            return torch.mean(torch.pow(x-target, 2)) * 1/torch.pow(self.sigma, 2)
        else:
            return torch.pow(x-target, 2) * 1/torch.pow(self.sigma, 2)

def guassian_nll(y_real, y_mu, rho, predict=False):
    var = torch.log(1+torch.exp(rho))
    if predict:
        return torch.sum(torch.pow(y_real-y_mu, 2) / var, dim=1).squeeze()
    return torch.sum(torch.pow(y_real-y_mu, 2) / var)

def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)