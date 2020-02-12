import torch
import torch.nn.functional as F


def initialize(model, init_type="xavier_uniform"):
    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()


class Net(torch.nn.Module):
    def __init__(self, idim, odim):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(idim, 32)
        self.fc2 = torch.nn.Linear(32, odim)

        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    def forward(self, x, y):
        # simple neural network for inverse network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # calculate loss with mse
        loss = F.mse_loss(input=x, target=y, reduction='mean')

        return loss
