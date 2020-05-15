import numpy as np
import torch

def mixup_data(x, y, ilens, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    x_device = x.device
    y_device = y.device

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = []
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    x_b = np.zeros(x.shape)
    y_b = np.zeros(y.shape)

    for i in range(batch_size):
        shuffle_range = list(range(ilens[i]))
        np.random.shuffle(shuffle_range)
        index.append(shuffle_range)
        x_b[i, :ilens[i]] = x[i, index[i]]
        y_b[i, :ilens[i]] = y[i, index[i]]

    mixed_x = torch.Tensor(lam * x + (1 - lam) * x_b)
    mixed_x = mixed_x.cuda(x_device)

    y_b = torch.Tensor(y_b).long().cuda(y_device)
    y = torch.Tensor(y).long().cuda(y_device)
    return mixed_x, y, y_b, index, lam

def mixup_logit(y, ilens, index, lam):
    '''Compute the mixup logit'''
    device = y.device
    batch_size = y.size()[0]

    y = y.data.cpu().numpy()
    y_b = np.zeros(y.shape)

    for i in range(batch_size):
        y_b[i, :ilens[i]] = y[i, index[i]]
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y_b)
    mixed_y = mixed_y.cuda(device)
    return mixed_y

def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch,
                                   total_steps_in_epoch, consistency_rampup_starts, consistency_rampup_ends):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * sigmoid_rampup(epoch, consistency_rampup_ends - consistency_rampup_starts)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
