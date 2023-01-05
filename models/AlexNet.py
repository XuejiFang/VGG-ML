import torch
import torch.nn as nn
from torchsummary import summary

# 原论文使用 2 个GPU并行运算，这里只用 1 个GPU，因此对通道数*2
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48 * 2, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
                        
            nn.Conv2d(48 * 2, 128 * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(128 * 2, 192 * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(192 * 2, 192 * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(192 * 2, 128 * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(256 * 6 * 6, 2048 * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
                        
            # nn.Dropout(p = 0.5),
            nn.Linear(2048 * 2, 2048 * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
                        
            nn.Linear(2048 * 2, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)