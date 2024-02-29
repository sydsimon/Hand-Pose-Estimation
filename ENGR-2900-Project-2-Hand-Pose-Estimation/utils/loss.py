import torch
from torch import nn


class Heatmap_Loss(nn.Module):
    """
    Modified dice loss between predicted and ground truth heatmap
    """
    def __init__(self):
        super(Heatmap_Loss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred_hm, gt_hm, joints_weight):
        """
        Calculate the dice loss between predicted and gt heatmaps ~ (B,K,H,W) 
        for all valid joints from B images where:
        B is batch size; K is number of joints; H and W are heatmap spatial dimension
        Input:
            pred_hm: (B,K,H,W) predicted heatmaps
            gt_hm: (B,K,H,W) ground truth heatmaps
            joints_weight: (B,K) valid flag, consisting of True and False
        Return:
            loss: (1,) modified dice loss value
        """
        assert len(pred_hm.shape) == 4, "Only works with batches of heatmap (B,K,H,W)"
        overlap = torch.sum(pred_hm * gt_hm, (2,3))
        total = torch.sum(pred_hm**2, (2,3)) + torch.sum(gt_hm**2, (2,3))
        dice_coeff = (2 * overlap + self.eps) / (total + self.eps) * joints_weight
        loss = torch.mean(1 - dice_coeff)
        return loss
