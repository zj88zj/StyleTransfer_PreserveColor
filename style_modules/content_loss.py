import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        _, C_l, H_l, W_l = content_current.shape
        F_l = content_current.view(C_l, H_l*W_l)
        P_l = content_original.view(C_l, H_l*W_l)
        loss = (torch.sum((F_l - P_l)**2)) * content_weight
        
        return loss


