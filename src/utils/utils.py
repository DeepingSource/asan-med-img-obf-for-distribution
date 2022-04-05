import torch

def one_hot(input, num_classes):
    input = input.unsqueeze(1)
    target_shape = list(input.shape)
    target_shape[1] = num_classes
    return torch.zeros(target_shape).to(input.device).scatter_(1, input, 1.)