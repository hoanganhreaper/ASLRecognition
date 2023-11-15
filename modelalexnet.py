import torch
from torchvision import models
from torch import nn

class ModelAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(ModelAlexNet, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        
        #Remove the last layer (classifier) of AlexNet
        self.features = alexnet.features

        #Add your custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        