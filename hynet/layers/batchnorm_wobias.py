import torch
import torch.nn as nn

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.bias.requires_grad = False

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=True) # unbiased really need?
            n = input.numel() / input.size(1)
            with torch.no_grad():
                # self.running_mean = exponential_average_factor * mean\
                #     + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        if (self.running_mean.sum() != 0) or (self.bias.sum() != 0):
            raise ValueError("bias and running_mean something goes wrong : {} {}".format(self.running_mean.sum(), self.bias.sum()))

        return input


