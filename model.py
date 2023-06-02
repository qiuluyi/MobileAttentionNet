import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, cls_num=2):
        super(MyModel, self).__init__()
        self.encode1 = nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            # torch.nn.MaxPool2d(2),
        )
        self.encode2 = nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            torch.nn.MaxPool2d(2),
        )
        self.encode3 = nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            torch.nn.MaxPool2d(2),
        )

        # self.encode1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, 2)
        # )
        # self.encode2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, 2)
        # )
        # self.encode3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, 2)
        # )

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.classifier = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        out = self.encode1(x)
        out = self.encode2(out)
        out = self.encode3(out)
        out = self.decode1(out)
        out = self.decode2(out)
        out = self.decode3(out)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)

    x = torch.rand(1, 3, 1080, 1920).to(device)
    y = model(x)
    print(x.shape)
    print(y.shape)

    flag = "Over !"
    print(flag)
