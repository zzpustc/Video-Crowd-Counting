from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *

# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class CascadeCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(CascadeCNN, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])
#        self.features5 = nn.Sequential(*features[0:23])


        self.de_pred_f = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))


        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 11 * 16, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc_1 = nn.Sequential(
            nn.Linear(10 * 11 * 16, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.localization_1 = nn.Sequential(
            nn.Conv2d(512, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.reload_weight()

        self.features5 = self.features4
        self.de_pred_p = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'),
                                       Conv2d(128, 1, 1, same_padding=True, NL='relu'))
#        self.de_pred_p  = self.de_pred_f                           
        self.freeze_f()


    def build_loss(self, density_map, gt_data):
        density_map = F.upsample(density_map,scale_factor=8)
        loss_mse = nn.MSELoss()(density_map.squeeze(), gt_data.squeeze())  
        return loss_mse


    def reload_weight(self):
        pretrained_dict = torch.load(cfg.Pre_MODEL_PATH)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_f(self):
        for param in self.features4.parameters():
            param.requires_grad = False
        for param in self.de_pred_f.parameters():
            param.requires_grad = False
        for param in self.features5.parameters():
            param.requires_grad = True
        for param in self.de_pred_p.parameters():
            param.requires_grad = True
        for param in self.localization.parameters():
            param.requires_grad = True
        for param in self.fc_loc.parameters():
            param.requires_grad = True
        for param in self.localization_1.parameters():
            param.requires_grad = True
        for param in self.fc_loc_1.parameters():
            param.requires_grad = True


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10*11*16) # for Mall, it is 10*11*16, for FDST it's 10*7*16
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def stn_early(self, x):
        xs = self.localization_1(x)
        xs = xs.view(-1, 10*11*16) # for Mall, it is 10*11*16, for FDST it's 10*7*16
        theta = self.fc_loc_1(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, x_p):
        x_p = self.features4(x_p)
        x_p1 = x_p
        x_p = self.de_pred_f(x_p)
        if cfg.STN == True:
           x_p = self.stn(x_p)
           x_p1 = self.stn_early(x_p1)

        x = self.features4(x) 
        x = torch.cat((x,x_p1),1)
        x = self.de_pred_p(x)

        x = x + x_p
        x = F.upsample(x,scale_factor=8)

        return x
