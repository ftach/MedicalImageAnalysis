import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv3d(128, 64, 1, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv3d(128, 256, 3, padding=1, padding_mode='reflect')
        self.conv6 = nn.Conv3d(128, 256, 1, padding=1, padding_mode='reflect')
        self.conv7 = nn.Conv3d(256, 128, 1, padding=1, padding_mode='reflect')
        self.conv8 = nn.Conv3d(256, 512, 3, padding=1, padding_mode='reflect')
        self.conv9 = nn.Conv3d(512, 256, 1, padding=1, padding_mode='reflect')
        self.conv10 = nn.Conv3d(512, 1024, 3, padding=1,
                                padding_mode='reflect')
        self.conv11 = nn.Conv3d(1024, 512, 1, padding=1,
                                padding_mode='reflect')
        self.conv12 = nn.Conv3d(1024, 2, 1, padding=1, padding_mode='reflect')
        self.maxpool = nn.MaxPool3d(2, 2, 0)
        self.globpool = nn.AvgPool3d(1)
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.batchnorm3 = nn.BatchNorm3d(128)
        self.batchnorm4 = nn.BatchNorm3d(256)
        self.batchnorm5 = nn.BatchNorm3d(512)
        self.batchnorm6 = nn.BatchNorm3d(1024)
        self.relu = nn.ReLU()  # scaled relu
        self.fc1 = nn.Linear(2, 128)

    def forward(self, x):
        x = self.maxpool(0.1*F.relu(self.batchnorm1(self.conv1(x))))  # 1-32
        x = self.maxpool(0.1*F.relu(self.batchnorm2(self.conv2(x))))  # 32-64
        x = 0.1*F.relu(self.batchnorm3(self.conv3(x)))  # 64-128
        x = 0.1*F.relu(self.batchnorm2(self.conv4(x)))  # 128-64, x, 1
        # 64-128, x, 3
        x = self.maxpool(0.1*F.relu(self.batchnorm3(self.conv3(x))))
        x = 0.1*F.relu(self.batchnorm4(self.conv5(x)))  # 128-256, 3
        x = 0.1*F.relu(self.batchnorm3(self.conv7(x)))  # 256-128, x, 1
        x = self.maxpool(
            0.1*F.relu(self.batchnorm4(self.conv5(x))))  # 128-256, 3
        x = 0.1*F.relu(self.batchnorm5(self.conv8(x)))  # 256-512, x, 3
        x = 0.1*F.relu(self.batchnorm4(self.conv9(x)))  # 512-256, x, 1
        x = 0.1*F.relu(self.batchnorm5(self.conv8(x)))  # 256-512, x, 3
        x = 0.1*F.relu(self.batchnorm4(self.conv9(x)))  # 512-256, x, 1
        x = self.maxpool(
            0.1*F.relu(self.batchnorm5(self.conv8(x))))  # 256-512,  3
        x = 0.1*F.relu(self.batchnorm6(self.conv10(x)))  # 512-1024, x, 3
        x = 0.1*F.relu(self.batchnorm5(self.conv11(x)))  # 1024-512, x, 1
        x = 0.1*F.relu(self.batchnorm6(self.conv10(x)))  # 512-1024, x, 3
        x = 0.1*F.relu(self.batchnorm5(self.conv11(x)))  # 1024-512, x, 1
        x = 0.1*F.relu(self.batchnorm6(self.conv10(x)))  # 512-1024, x, 3
        x = self.globpool(self.conv12(x))  # 128 features
        x = x.view(-1, 2)
        x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


def model_summary(model, input_size):
    def forward_hook(module, input, output):
        layer_name = str(module)
        num_params = sum(p.numel() for p in module.parameters())
        print(
            f"{layer_name:20} | Input shape: {str(input[0].shape):30} | Output shape: {str(output.shape):30} | Parameters: {num_params}")

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    print(f"{'Layer':20} | {'Input Shape':30} | {'Output Shape':30} | {'Param #'}")
    print("="*95)

    try:
        model(torch.rand(1, *input_size))
    finally:
        for hook in hooks:
            hook.remove()


# input
# 3D Conv 3x3x3, L channels, stride=1, padding=reflect (L will change)
# batch norm , L channels
# Relu scale=0.1
# Max-Pooling filter=2,2,2 stride=2,2,2 padding=zero
#
net = RegNet()
depth = 32  # number of slices used
model_summary(net, (1, depth, 256, 256))
