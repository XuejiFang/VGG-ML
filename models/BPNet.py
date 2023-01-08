import torch
import torch.nn as nn

class BPNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return x
    
def bpnet(num_classes): 
    model = BPNet(num_classes=num_classes)
    return model