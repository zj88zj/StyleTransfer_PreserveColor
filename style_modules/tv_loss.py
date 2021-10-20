import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        variance_x = torch.sum((img[:,:,:,1:] - img[:,:,:,:-1])**2)
        variance_y = torch.sum((img[:,:,1:,:] - img[:,:,:-1,:])**2)
        loss = (variance_x + variance_y) * tv_weight
        
        return loss