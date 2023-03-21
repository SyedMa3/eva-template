import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.X = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3,
                    stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        y = self.X(x)
        out = F.relu(self.bn1(self.conv1(y)))
        out = self.bn2(self.conv2(out))
        out += y
        return out


class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,
                    stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        self.in_planes = 256
        out = self.layer3(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return F.log_softmax(out, dim=-1)


def MakeResNet():
    return CustomResNet(BasicBlock, [1, 1, 1])


def test():
    net = MakeResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
