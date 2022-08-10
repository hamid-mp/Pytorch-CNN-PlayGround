import torch
import torch.nn as nn
from layers import Block
import torch.nn.functional as F
#from torchviz import make_dot



class Model(nn.Module):
    def __init__(self):
        
        super(Model, self).__init__()
        self.conv2d1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,padding=1, stride=2),
        nn.BatchNorm2d(8),
        nn.ReLU6(inplace=True))

        self.stage1 = nn.Sequential(
            Block(8, 8, exp=1, stride=2, type='C'))

        self.stage2 = nn.Sequential(
            Block(8, 16, exp=2, stride=2, type='C'),
            Block(16, 16, exp=2, type='A'))
        self.stage3 = nn.Sequential(
            Block(16, 24, exp=2, stride=2, type='C'),
            Block(24, 24, exp=2, type='A'))
        self.post_block2 = nn.Sequential(
            Block(24, 32, exp=2, type='B'))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout()
        self.head =nn.Sequential(
        nn.Linear(32, 10)) 

    def forward(self, x):
        out = self.conv2d1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.post_block2(out)
        out = self.gap(out)
        out = out.view(-1, 32)
        out = self.drop(out)

        out = self.head(out)
        return out
# ---------------- Visualize model architecture with dummpy input
#x = torch.randn(50,1,32, 32)
#model =Model()
#yhat = model(x)

#make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
#input_names = ['Sentence']
#output_names = ['yhat']
#torch.onnx.export(model, x, 'rnn.onnx', input_names=input_names, output_names=output_names)