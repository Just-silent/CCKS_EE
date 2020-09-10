import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # shape(y_pred) = batch_size, label_num, **
        # shape(y_true) = batch_size, **
        y_pred = torch.softmax(y_pred, dim=1)
        pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
        dsc_i = 1 - ((1 - pred_prob) * pred_prob) / ((1 - pred_prob) * pred_prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss

# class DiceLoss(torch.nn.Module):
#     def init(self):
#         super(DiceLoss, self).init()
#
#     def forward(self, pred, target):
#         smooth = 1.
#         iflat = pred.contiguous().view(-1)
#         tflat = target.contiguous().view(-1)
#         intersection = (iflat * tflat).sum()
#         A_sum = torch.sum(iflat * iflat)
#         B_sum = torch.sum(tflat * tflat)
#         return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()