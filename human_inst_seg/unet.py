import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# this can be installed by:
# pip install git+https://github.com/liruilong940607/human_det --upgrade
from human_det import Detection

# this can be installed by:
# pip install git+https://github.com/qubvel/segmentation_models.pytorch --upgrade
import segmentation_models_pytorch as smp

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
def scale_boxes(boxes, scale):
    """
    Args:
        boxes (tensor): A tensor of shape (B, 4) representing B boxes with 4
            coords representing the corners x0, y0, x1, y1,
        scale (float, float): The box scaling factor (w, h).
    Returns:
        Scaled boxes.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale[0]
    h_half *= scale[1]

    scaled_boxes = torch.zeros_like(boxes)
    scaled_boxes[:, 0] = x_c - w_half
    scaled_boxes[:, 2] = x_c + w_half
    scaled_boxes[:, 1] = y_c - h_half
    scaled_boxes[:, 3] = y_c + h_half
    return scaled_boxes

class Segmentation(nn.Module):
    def __init__(self, ckpt=None, device="cuda:0", init=True):
        super().__init__()
        model = smp.Unet(
            'resnet18',
            encoder_weights='imagenet', 
            classes=2, 
            # activation='softmax'
        ).to(device)

        if ckpt is not None and os.path.exists(ckpt):
            print (f"load ckpt from: {ckpt}")
            model.load_state_dict(torch.load(ckpt))
        
        self.device = device
        self.model = model

        self.det_engine = Detection(device=device)
        
        if init:
            self.init()

    def init(self, pretrained=""):
        if os.path.exists(pretrained):
            state_dict = torch.load(pretrained)
        else:
            state_dict = load_state_dict_from_url(
                "https://drive.google.com/uc?export=download&id=18d2yeCx62Gup-YzgsI866uxpEo9kIl2T")
        self.load_state_dict(state_dict)

    def forward(self, input):  
        # input is 1 x 3 x H x W
        Batch, _, H, W = input.size() 
        input = input.to(self.device)

        # det
        with torch.no_grad(): 
            bboxes_det, probs_det = self.det_engine(input)
            
            probs = probs_det.unsqueeze(3)
            bboxes = (bboxes_det * probs).sum(dim=1, keepdim=True) / probs.sum(dim=1, keepdim=True)
            bboxes = bboxes[:, 0, 0, :]
            
            w_half = (bboxes[:, 2] - bboxes[:, 0]) * 0.5
            h_half = (bboxes[:, 3] - bboxes[:, 1]) * 0.5
            x_c = (bboxes[:, 2] + bboxes[:, 0]) * 0.5
            y_c = (bboxes[:, 3] + bboxes[:, 1]) * 0.5
            h_half *= 1.2 if not self.training else random.uniform(1.0, 1.5)
            w_half = h_half / 288 * 192
            scaled_boxes = torch.zeros_like(bboxes)
            scaled_boxes[:, 0] = x_c - w_half
            scaled_boxes[:, 2] = x_c + w_half
            scaled_boxes[:, 1] = y_c - h_half
            scaled_boxes[:, 3] = y_c + h_half    
            scaled_boxes = [box.unsqueeze(0) for box in scaled_boxes]

        # seg
        output = self.model(
            torchvision.ops.roi_align(input, scaled_boxes, (288, 192)))

        x0_int, y0_int = 0, 0
        x1_int, y1_int = W, H
        scaled_boxes = torch.cat(scaled_boxes, dim=0)
        x0, y0, x1, y1 = torch.split(scaled_boxes, 1, dim=1)  # each is Nx1

        img_y = torch.arange(y0_int, y1_int, device=self.device, dtype=torch.float32) + 0.5
        img_x = torch.arange(x0_int, x1_int, device=self.device, dtype=torch.float32) + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        # img_x, img_y have shapes (N, w), (N, h)

        gx = img_x[:, None, :].expand(Batch, img_y.size(1), img_x.size(1))
        gy = img_y[:, :, None].expand(Batch, img_y.size(1), img_x.size(1))
        grid = torch.stack([gx, gy], dim=3)

        # train.py
        if self.training:
            output = F.grid_sample(output, grid, align_corners=False)
            output = F.interpolate(output, size=(H, W), mode="bilinear")
            return output
        
        else:
            output = F.softmax(output, dim=1)[:, 1:2]
            output = F.grid_sample(output, grid, align_corners=False)
            output = F.interpolate(output, size=(H, W), mode="bilinear")  
            output = (output > 0.5).float()
            output = torch.cat([input, output], dim=1)
            return output, bboxes_det, probs_det

