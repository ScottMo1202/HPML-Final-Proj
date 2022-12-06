import torch

# The model is based on the website https://blog.paperspace.com/alexnet-pytorch/.

class AlexNet4(torch.nn.Module):
    def __init__(self, input_channel, num_classes=2):
        super(AlexNet4, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=0),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU())
        self.fc2= torch.nn.Sequential(
            torch.nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def alexnet():
    return AlexNet4(input_channel=3, num_classes=2)
