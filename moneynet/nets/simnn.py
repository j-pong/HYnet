import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, idim, odim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(idim, 50)
        self.fc2 = nn.Linear(50, odim)

    def forward(self, x, y):
        # transform the input
        x = self.stn(x)

        # simple neural network for inverse network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # calculate loss with mse
        loss = F.mse_loss(input=x, target=y, reduction='mean')

        return loss
