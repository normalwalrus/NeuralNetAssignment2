import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self, dropout = 0.25, kernel_size = 3):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.drop = nn.Dropout2d(dropout)
        self.fc3 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
class CNN_no_batch(nn.Module):
    
    def __init__(self, dropout = 0.25, kernel_size = 3):
        super(CNN_no_batch, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.drop = nn.Dropout2d(dropout)
        self.fc3 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
class CNN_no_maxpool(nn.Module):
    
    def __init__(self, dropout = 0.25, kernel_size = 3):
        super(CNN_no_maxpool, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(in_features=64*26*26, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.drop = nn.Dropout2d(dropout)
        self.fc3 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
class CNN_no_maxpool_batchnorm(nn.Module):
    
    def __init__(self, dropout = 0.25, kernel_size = 3):
        super(CNN_no_maxpool_batchnorm, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(in_features=64*26*26, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.drop = nn.Dropout2d(dropout)
        self.fc3 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    

class CNN_kernel_check(nn.Module):
    
    def __init__(self, dropout = 0.25, kernel_size = 3):
        super(CNN_kernel_check, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=576, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.drop = nn.Dropout2d(dropout)
        self.fc3 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
class CNN_conv_layer(nn.Module):
    
    def __init__(self, dropout = 0.25, kernel_size = 3):
        super(CNN_conv_layer, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=6272, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.drop = nn.Dropout2d(dropout)
        self.fc3 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


class CNN_fully_connected(nn.Module):
    
    def __init__(self, kernel_size = 3, dropout = 0.25):
        super(CNN_fully_connected, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.fc3 = nn.Linear(in_features=300, out_features=300)
        self.fc4 = nn.Linear(in_features=300, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.drop(out)
        # out = self.fc3(out)
        # out = self.drop(out)
        out = self.fc4(out)
        
        return out
    