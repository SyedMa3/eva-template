import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformer(nn.Module):
    def __init__(self):
        super(CustomTransformer, self).__init__()

        val = 0.05

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(val),

            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(val),

            nn.Conv2d(32, 48, 3, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(val)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.U = nn.Sequential(
            UltimusBlock(),
            UltimusBlock(),
            UltimusBlock(),
            UltimusBlock()
        )

        self.fc = nn.Linear(48, 10, False)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)

        x = x.view(-1, 48)

        x = self.U(x)

        out = self.fc(x)

        out = out.view(-1, 10)
        return out


class UltimusBlock(nn.Module):
    def __init__(self):
        super(UltimusBlock, self).__init__()

        self.k = nn.Linear(48,48, False)
        self.q = nn.Linear(48,48, False)
        self.v = nn.Linear(48,48, False)

        self.out = nn.Linear(48,48, False)

    def forward(self, x):

        XK = self.k(x)
        XK = XK.view(XK.size(0), 1, -1)
        XQ = self.q(x)
        XQ = XQ.view(XQ.size(0), -1, 1)
        XV = self.v(x)
        XV = XV.view(XV.size(0), 1, -1)

        AM = self.amul(XQ, XK)

        Z = torch.matmul(XV, AM)

        out = self.out(Z)

        return out

    def amul(self, x, y):

        am = torch.matmul(x.transpose(1,2),y) / (8**0.5)

        return F.softmax(am, dim=1)
