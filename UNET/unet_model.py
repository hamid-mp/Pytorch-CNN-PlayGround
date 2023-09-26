import torch
import torch.nn as nn




class DoubleConv(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(DoubleConv, self).__init__()

        self.c1 = nn.Sequential(nn.Conv2d(input_channel, out_channel,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(out_channel, out_channel,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU())

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x

class UNET(nn.Module):

    def __init__(self, input_c, num_cls):
        super(UNET, self).__init__()

        self.input_channels = input_c
        self.class_num = num_cls


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Encoder
        self.down_conv1 = DoubleConv(self.input_channels, 64)
        self.down_conv2 = DoubleConv(64,128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.down_conv5 = DoubleConv(512, 1024)


        # Decoder
        self.up_conv1 = DoubleConv(1024, 512)
        self.up_conv2 = DoubleConv(512, 256)
        self.up_conv3 = DoubleConv(256, 128)
        self.up_conv4 = DoubleConv(128, 64)
        
        #head
        self.conv1x1 = nn.Conv2d(64, self.class_num, 1, stride=1, padding=0)

        # Decoder part / Expansion phase
        self.conv_trans1=nn.ConvTranspose2d(in_channels=1024,
                                           out_channels=512,
                                           kernel_size=2,
                                           stride=2)

        self.conv_trans2=nn.ConvTranspose2d(in_channels=512,
                                           out_channels=256,
                                           kernel_size=2,
                                           stride=2)


        self.conv_trans3=nn.ConvTranspose2d(in_channels=256,
                                           out_channels=128,
                                           kernel_size=2,
                                           stride=2)


        self.conv_trans4=nn.ConvTranspose2d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=2,
                                           stride=2)


    def forward(self, x):
        x1 = self.down_conv1(x)
        x = self.maxpool(x1)

        x2 = self.down_conv2(x)
        x = self.maxpool(x2)

        x3 = self.down_conv3(x)
        x = self.maxpool(x3)

        x4 = self.down_conv4(x)
        x = self.maxpool(x4)

        x = self.down_conv5(x)

        # Decoder Part
        #block 1
        x = self.conv_trans1(x)
        cat1 = torch.cat(tensors=(x, x4), dim=1)
        x = self.up_conv1(cat1)
        
        # block 2

        x = self.conv_trans2(x)
        cat2 = torch.cat(tensors=(x, x3), dim=1)
        x = self.up_conv2(cat2)

        # block 3

        x = self.conv_trans3(x)
        cat3 = torch.cat(tensors=(x, x2), dim=1)
        x = self.up_conv3(cat3)

        # block 4

        x = self.conv_trans4(x)
        cat4 = torch.cat(tensors=(x, x1), dim=1)
        x = self.up_conv4(cat4)
        x = self.conv1x1(x)
        return x
