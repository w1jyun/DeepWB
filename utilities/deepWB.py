"""
 Deep white-balance editing main function (inference phase)
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
import torch
from torchvision import transforms
import utilities.utils as utls

import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

def deep_wb(img, net_awb=None, net_t=None, net_s=None, device='cpu', s=656):
    _, _, h, w = img.shape
    max_side = max(h, w)
    blur = GaussianBlur(kernel_size=5, sigma=1.0)
    # 리사이즈 (비율 유지, 한쪽이 s에 맞춰짐)
    new_w = round(w / max_side * s)
    new_h = round(h / max_side * s)
    img_resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # 16 배수로 패딩
    pad_h = (16 - new_h % 16) % 16
    pad_w = (16 - new_w % 16) % 16
    image_resized = F.pad(img_resized, (0, pad_w, 0, pad_h), mode="reflect")
    image_resized = blur(image_resized) 
    
    net_awb.eval()
    with torch.no_grad():
        output_awb = net_awb(image_resized)
    
    m_awb = utls.get_mapping_func_t(image_resized, output_awb)
    output_awb = utls.out_of_gamut_clipping_t(utls.apply_mapping_func_t(img, m_awb))
    return output_awb

