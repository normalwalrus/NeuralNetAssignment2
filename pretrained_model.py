import torchvision.models as models
import torch
import torch.nn as nn

class ResNet_pretrained(nn.Module):
  def __init__(self, model_choice = 'resnet18', in_channels=1, train_last_layer_only = False):
    super(ResNet_pretrained, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    if model_choice == 'resnet18':
        self.model = models.resnet18(weights='IMAGENET1K_V1')
    elif model_choice == 'resnet50':
        self.model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

    if train_last_layer_only:
      for param in self.model.parameters():
        param.requires_grad = False

    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)

  def forward(self, x):
    return self.model(x)
  
class vgg16_pretrained(nn.Module):
  def __init__(self, in_channels=1, train_last_layer_only = False):
    super(vgg16_pretrained, self).__init__()
    self.model = models.vgg16(pretrained = True)
    
    if train_last_layer_only:
      for param in self.model.parameters():
        param.requires_grad = False

    first_conv_layer = [nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(self.model.features))  
    self.model.features= nn.Sequential(*first_conv_layer )  
    
    self.layer1 = nn.Linear(1000,10)

  def forward(self, x):

    output = self.model(x)
    output = self.layer1(output)
    return output