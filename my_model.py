import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=1,do_batch=1):
        super().__init__()

        self.do_batch=do_batch

        self.conv=nn.Conv2d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)
        self.dropout=nn.Dropout2d(0.1)



    def forward(self, inputs):
        outputs = self.conv(inputs)

        if self.do_batch:
            outputs = self.bn(outputs)

        outputs=F.relu(outputs)
        outputs=self.dropout(outputs)

        return outputs




class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=2,pad=1,out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_size, out_size,filter_size,stride=stride, padding=pad, output_padding=out_pad)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs=F.relu(outputs)
        return outputs




class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()

        self.up = unetConvT2(in_size, out_size )

        self.dropout = nn.Dropout2d(0.1)  # Dropout v upsampling části

    def forward(self, inputs1, inputs2):

        inputs2 = self.up(inputs2)

        ### pading for nondivisible sizes of image
        shape1=list(inputs1.size())
        shape2=list(inputs2.size())

        pad=(0,shape1[-1]-shape2[-1],0,shape1[-2]-shape2[-2])

        inputs2=F.pad(inputs2,pad)

        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.dropout(outputs)  # Dropout po spojení feature map
        
        return outputs




class Unet2D(nn.Module):
    def __init__(self, filters=(np.array([16, 32, 64, 128])).astype(int),in_size=1,out_size=1):
        super().__init__()

        self.out_size = out_size

        self.in_size = in_size

        self.filters = filters


        self.conv1 = nn.Sequential(unetConv2(in_size, filters[0]),unetConv2(filters[0], filters[0]),unetConv2(filters[0], filters[0]))

        self.conv2 =  nn.Sequential(unetConv2(filters[0], filters[1] ),unetConv2(filters[1], filters[1] ),unetConv2(filters[1], filters[1] ))

        self.conv3 = nn.Sequential(unetConv2(filters[1], filters[2] ),unetConv2(filters[2], filters[2] ),unetConv2(filters[2], filters[2] ))


        self.center = nn.Sequential(unetConv2(filters[-2], filters[-1] ),unetConv2(filters[-1], filters[-1] ))



        self.up_concat3 = unetUp(filters[3], filters[3] )
        self.up_conv3=nn.Sequential(unetConv2(filters[2]+filters[3], filters[2] ),unetConv2(filters[2], filters[2] ))



        self.up_concat2 = unetUp(filters[2], filters[2] )
        self.up_conv2=nn.Sequential(unetConv2(filters[1]+filters[2], filters[1] ),unetConv2(filters[1], filters[1] ))


        self.up_concat1 = unetUp(filters[1], filters[1])
        self.up_conv1=nn.Sequential(unetConv2(filters[0]+filters[1], filters[0] ),unetConv2(filters[0], filters[0],do_batch=0 ))



        self.final = nn.Conv2d(filters[0], self.out_size, 1)



        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                # init.xavier_normal_(m.weight)
                # init.constant_(m.bias, 0)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)



    def forward(self, inputs):


        conv1 = self.conv1(inputs)
        x = F.max_pool2d(conv1,2,2)


        conv2 = self.conv2(x)
        x = F.max_pool2d(conv2,2,2)


        conv3 = self.conv3(x)
        x = F.max_pool2d(conv3,2,2)



        x = self.center(x)




        x = self.up_concat3(conv3, x)
        x = self.up_conv3(x)


        x = self.up_concat2(conv2, x)
        x=self.up_conv2(x)

        x = self.up_concat1(conv1, x)
        x=self.up_conv1(x)


        x = self.final(x)

        return x