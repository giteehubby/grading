
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
import torch

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=strides)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3 = None
        if strides != 1:
            self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=strides)

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)

class resnet_block(nn.Module):

    def __init__(self,input_channels, num_channels, num_residuals=3,
                 first_block=False):
        super(resnet_block,self).__init__()
        self.num_residual = num_residuals
        for i in range(num_residuals):
            if i == 0 and not first_block:#通道和画幅只在first_block=False时改变
                setattr(self, f'block{i}', Residual(input_channels, num_channels,strides=2))
            else:
                setattr(self, f'block{i}', Residual(num_channels, num_channels))

    def forward(self,X):
        for i in range(self.num_residual):
            residual = getattr(self,f'block{i}')
            X = residual(X)
        return X

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        # input : batch_size,3,512,512
        self.blks = nn.Sequential(
            resnet_block(3,3,3,first_block=True),
            # 3, 512, 512
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            # 3, 256, 256
            resnet_block(3,64,3),
            # 64, 128, 128
            resnet_block(64,128,2),
            # 128, 64, 64
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            # 128, 32, 32
            resnet_block(128,256,2),
            # 256, 16, 16
            resnet_block(256,512,2),
            # 512, 8, 8
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            # 512, 4, 4
            resnet_block(512,512,2),
            # 256, 2, 2
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # nn.Dropout(0.7),
            nn.Linear(512,64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64,2),
        )

    def forward(self,X):
        for blk in self.blks:
            if isinstance(blk,nn.Sequential):
                for b in blk:
                    X = b(X)
            else:
                X = blk(X)
        return X

# device = torch.device('cuda:0')
# resnet = Resnet()
# resnet.to(device)
# X = torch.randn(5,3,512,512)
# X = X.to(device)
# y = resnet(X)
# summary(resnet, (3, 512, 512), device='cuda')
# print(y.shape)