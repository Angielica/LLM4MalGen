import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()

        self.criterion = nn.BCELoss()

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        return loss

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x_rec, x,  mu, log_var, beta=1.0):
        mse_loss = self.criterion(x_rec, x)
        kld_loss = self.kld(mu, log_var)

        return mse_loss + beta * kld_loss, mse_loss, kld_loss

    def kld(self, mu, log_var):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        return kld