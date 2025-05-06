import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

#our custom loss
def Anomaly_VAE_loss(output, target):
    output = output[0].view(target.shape)
    loss = F.mse_loss(output, target)
    return loss
