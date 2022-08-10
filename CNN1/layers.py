
import torch.nn as nn
import torch
import torch.nn.functional as F
    
class Linear_activation(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, exp=1, stride=1, type=''):
        super(Block, self).__init__()
        self.t = type
        self.stride = stride
        self.inc, self.outc = in_channels, out_channels
        self.exp = exp

        
        self.blockc = nn.Sequential(
            nn.Conv2d(self.inc, self.inc* self.exp, kernel_size=1),
            nn.BatchNorm2d(self.inc * self.exp),
            

            nn.ReLU6(inplace=True), 
            nn.Conv2d(self.inc * self.exp, self.inc * self.exp, kernel_size=3, groups= self.inc * self.exp, stride= self.stride, padding=1),
            nn.BatchNorm2d(self.inc * self.exp),
            nn.ReLU6(inplace=True),
            nn.Conv2d(self.inc * self.exp, self.outc, kernel_size=1),
        
            nn.BatchNorm2d(self.outc))
    def forward(self, x):
        out = self.blockc(x)
        
        if self.t == 'A':
            out = torch.add(out,x)

        return out
