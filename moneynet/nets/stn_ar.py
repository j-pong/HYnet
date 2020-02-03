import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, idim, odim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(idim, 50)
        self.fc2 = nn.Linear(50, odim)

    # Spatial transformer network forward function
    def stn(self, x):
        # xs = self.localization(x)
        # xs = xs.view(-1, 10 * 3 * 3)
        # theta = self.fc_loc(xs)
        # theta = theta.view(-1, 2, 3)
        #
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)

        return x

    def forward(self, x, y):
        # transform the input
        x = self.stn(x)

        # simple neural network for inverse network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # calculate loss with mse
        loss = F.mse_loss(input=x, target=y, reduction='mean')

        return loss
