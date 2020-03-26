#check if crop cat works

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downc1 = downStep(1,64)
        self.mpool1 = nn.MaxPool2d(2,2)
        self.downc2 = downStep(64,128)
        self.mpool2 = nn.MaxPool2d(2,2)
        self.downc3 = downStep(128,256)
        self.mpool3 = nn.MaxPool2d(2,2)
        self.downc4 = downStep(256,512)
        self.mpool4 = nn.MaxPool2d(2,2)
        
        self.conv1 = nn.Conv2d(512, 1024, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1024, 3)
        self.relu2 = nn.ReLU()
        
        self.upc4 = upStep(1024, 512)
        self.upc3 = upStep(512, 256)
        self.upc2 = upStep(256, 128)
        self.upc1 = upStep(128, 64)
        
        self.conv_final = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        '''
        x1 = self.downc1(x)
        x2 = self.downc2(F.max_pool2d(x1,2,2))
        x3 = self.downc3(F.max_pool2d(x2,2,2))
        x4 = self.downc4(F.max_pool2d(x3,2,2))
        '''
        x1 = self.downc1(x)
        x2 = self.mpool1(x1)
        x2 = self.downc2(x2)
        x3 = self.mpool2(x2)
        x3 = self.downc3(x3)
        x4 = self.mpool3(x3)
        x4 = self.downc4(x4)
        x5 = self.mpool4(x4)
        
        x = self.conv1(x5)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.upc4(x, x4)
        x = self.upc3(x, x3)
        x = self.upc2(x, x2)
        x = self.upc1(x, x1, withReLU=False)
        
        x = self.conv_final(x)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # Do not forget to concatenate with respective step in contracting path
        self.deconv = nn.ConvTranspose2d(inC, outC, 2, stride=2) # Confirm if stride 2
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.relu2 = nn.ReLU()
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!

    def forward(self, x, x_down, withReLU=True):
        x = self.deconv(x)
        
        #crop
        prev_size = x_down.size()
        new_size = x.size()
        no_w = int((prev_size[2]-new_size[2])/2)
        no_h = int((prev_size[3]-new_size[3])/2)
        x_down = x_down[:, :, no_w:(new_size[2]+no_w), no_h:(new_size[3]+no_h)]
        
        #concatenate
        x = torch.cat((x_down, x), 1) #x_down, x
        
        x = self.conv1(x)
        if withReLU==True:
            x = self.relu1(x)
        
        x = self.conv2(x)      
        if withReLU==True:
            x = self.relu2(x)
        return x