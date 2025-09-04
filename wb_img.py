import argparse
import logging
import os
import torch
from PIL import Image
from arch import deep_wb_model
import utilities.utils as utls
from utilities.deepWB import deep_wb
import arch.splitNetworks as splitter
from arch import deep_wb_single_task
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn

class DeepWB(nn.module):
    def __init__(self, model_dir: str = "./models"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load awb net
        net_awb = deep_wb_single_task.deepWBnet()
        net_awb.to(device=self.device)
        net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                        map_location=self.device))
        net_awb.eval()
        self.net = net_awb

    def forward(self, img):
        out_awb = deep_wb(img, net_awb=self.net, device=self.device)
        return out_awb