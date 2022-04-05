import torch

from utils.loss import compute_per_channel_dice


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        self.input = None
        self.target = None

    def reset(self):
        self.input = None
        self.target = None

    def update(self, input, target):
        input = input.detach().cpu()
        target = target.detach().cpu()
        if self.input == None:
            self.input = input
        else:
            self.input = torch.cat((self.input, input), axis=0)
        if self.target == None:
            self.target = target
        else:
            self.target = torch.cat((self.target, target), axis=0)

    def compute(self):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(self.input, self.target, epsilon=self.epsilon))

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.input)
        torch.distributed.all_reduce(self.target)

    def __str__(self):
        dice_coeff = self.compute()
        return ('dice coefficient: {:.4f}\t').format(dice_coeff.item())

# Abstract class
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat:
            self.mat.zero_()

    def compute(self):
        pass

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        pass


class ConfusionMatrixAsan(ConfusionMatrix):
    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0))
        return acc_global, acc, iu, dice

    def __str__(self):
        acc_global, acc, iu, dice = self.compute()
        return ('global correct: {:.1f}\t'
                'average row correct: {}\t'
                'IoU: {}\t'
                'mean IoU: {:.1f}\t'
                'Dice: {}\t'
                'mean Dice: {:.1f}\t').format(
                    acc_global.item() * 100,
                    ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                    ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                    iu.mean().item() * 100,
                    ['{:.1f}'.format(i) for i in (dice * 100).tolist()],
                    dice.mean().item() * 100
                    )


class ConfusionMatrixBrats(ConfusionMatrix):
    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0))

        tp = h[:, [1, 2, 3, 4]][[1, 2, 3, 4], :].sum()
        fn = h[[1, 2, 3, 4], :][:, 0].sum()
        fp = h[0, :][[1, 2, 3, 4]].sum()
        dice_complete = 2 * tp / (2 * tp + fp + fn)

        tp = h[:, [1, 3, 4]][[1, 3, 4], :].sum()
        fn = h[[1, 3, 4], :][:, [0, 2]].sum()
        fp = h[[0, 2], :][:, [1, 3, 4]].sum()
        dice_core = 2 * tp / (2 * tp + fp + fn)

        tp = h[4][4].sum()
        fn = h[4][[0, 1, 2, 3]].sum()
        fp = h[[0, 1, 2, 3], :][:, 4].sum()
        dice_enhancing = 2 * tp / (2 * tp + fp + fn)

        return acc_global, acc, iu, dice, dice_complete, dice_core, dice_enhancing

    def __str__(self):
        acc_global, acc, iu, dice, dice_complete, dice_core, dice_enhancing = self.compute()
        return ('global correct: {:.1f}\t'
                'average row correct: {}\t'
                'IoU: {}\t'
                'mean IoU: {:.1f}\n'
                'Dice: {}\t'
                'total Dice: {:.1f}\t'
                'complete Dice: {:.1f}\t'
                'core Dice: {:.1f}\t'
                'enhancing Dice: {:.1f}\t').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
            ['{:.1f}'.format(i) for i in (dice * 100).tolist()],
            dice.mean().item() * 100,
            dice_complete * 100,
            dice_core * 100,
            dice_enhancing * 100)