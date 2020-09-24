import numpy as np
import torch
import torch.nn.functional as F

def mixup_data(x, y, ilens, alpha, scheme="global"):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    x_device = x.device
    y_device = y.device

    if alpha > 0.:
        lam = np.random.uniform(alpha, 1)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = []
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    x_b = np.zeros(x.shape)
    y_b = np.zeros(y.shape)

    if scheme == "local":
        for i in range(batch_size):
            shuffle_range = list(range(ilens[i]))
            np.random.shuffle(shuffle_range)
            index.append(shuffle_range)
            x_b[i, :ilens[i]] = x[i, index[i]]
            y_b[i, :ilens[i]] = y[i, index[i]]
        mixed_x = torch.Tensor(lam * x + (1 - lam) * x_b)
    elif scheme == "global":
        batch_size = x.shape[0]
        index = torch.randperm(batch_size)
        mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index, :])
        y_b = y[index]

    mixed_x = mixed_x.to(x_device)

    y_b = torch.Tensor(y_b).to(y_device).long()
    y = torch.Tensor(y).to(y_device).long()
    return mixed_x, y, y_b, index, lam

def reverse_mixup_data(x, y, ilens, alpha, scheme="global"):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    x_device = x.device
    y_device = y.device

    if alpha > 0.:
        lam = np.random.uniform(alpha, 1)
    else:
        lam = 1.

    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()

    batch_size = x.shape[0]
    index = torch.randperm(batch_size)
    mixed_x = torch.Tensor(lam * x + (1 - lam) * np.flip(x[index, :],1))
    y_b = y[index]

    mixed_x = mixed_x.to(x_device)

    y_b = torch.flip(torch.Tensor(y_b),[1]).to(y_device).long()
    y = torch.Tensor(y).to(y_device).long()
    return mixed_x, y, y_b, index, lam

def mixup_logit(y, ilens, index, lam, scheme="global"):
    '''Compute the mixup logit'''
    device = y.device
    batch_size = y.size()[0]

    y = y.data.cpu().numpy()
    y_b = np.zeros(y.shape)

    if scheme == "local":
        for i in range(batch_size):
            y_b[i, :ilens[i]] = y[i, index[i]]
    elif scheme == "global":
        y_b = y[index]
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y_b)
    mixed_y = mixed_y.to(device)
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

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def softmax_mse_loss(input_logits, target_logits, reduction_str=None):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax, reduction=reduction_str)
