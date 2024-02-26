import torch
import torch.nn as nn
from torchvision.datasets.folder import find_classes

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.conv64_7_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.maxpool1_2_2  = nn.MaxPool2d(2, stride=2, padding=0)
        #torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.conv192_3_1 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)


        self.conv128_1_1 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv256_3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv256_1_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv512_3_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv512_256_1_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv512_1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv1024_3_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.conv64_7_2(x)#conv 1
        out = self.maxpool1_2_2(out)
        print(out.size())


        out = self.conv192_3_1(out)#conv 2
        out = self.maxpool1_2_2(out)
        print(out.size())



        out = self.conv128_1_1(out)#Layer 3
        out = self.conv256_3_1(out)
        out = self.conv256_1_1(out)
        out = self.conv512_3_1(out)
        out = self.maxpool1_2_2(out)
        print(out.size())


        out = self.conv512_256_1_1(out)
        out = self.conv512_3_1(out)
        out = self.conv512_256_1_1(out)
        out = self.conv512_3_1(out)
        out = self.conv512_256_1_1(out)
        out = self.conv512_3_1(out)
        out = self.conv512_256_1_1(out)
        out = self.conv512_3_1(out)
        out = self.conv512_1_1(out)
        out = self.conv1024_3_1(out)
        out = self.maxpool1_2_2(out)

        return out

img = torch.randn(1,3,448,448)
model = YOLO()
x = model(img)
print(x.size())
#print(find_classes("C://ILSVRC2012_img_train"))