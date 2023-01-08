import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ConvNet, self).__init__()
        # 定义卷积模块self_layer1, self_layer2
        self.layer1 = nn.Sequential(
            # 输入通道1，输出通道16，卷积核5*5，边衬2单位
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            # 批量归一化
            nn.BatchNorm2d(16),
            # 激活函数层
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        
        
    def forward(self, x):
        #print("x size:\t", x.shape)
        out = self.layer1(x)   #; print("out1 size:\t", out.shape)
        out = self.layer2(out) #; print("out2 size:\t", out.shape)
        out = out.reshape(out.size(0), -1) #; print("out3 size:\t", out.shape)
        out = self.fc(out)    #; print("outFinal size:\t", out.shape)

        return out