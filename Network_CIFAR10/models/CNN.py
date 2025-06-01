import torch
import torch.nn as nn

class LeNetLike(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(LeNetLike, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU(),
            # dropout layer
            nn.Dropout(p=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            # dropout layer
            nn.Dropout(p=0.1),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AdvLeNetLike(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(AdvLeNetLike, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=120, kernel_size=5),
            nn.BatchNorm2d(num_features=120),
            nn.ReLU(),

            nn.Dropout(p=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            # nn.BatchNorm1d(num_features=84),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        self.downsample = downsample

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        input_copy = x
        output = self.layers(input_copy)
        if self.downsample:
            input_copy = self.downsample(input_copy)
        output += input_copy
        return self.ReLU(output)


class SmallResNet(nn.Module):
    def __init__(self, input_channels, num_classes=10):
        super(SmallResNet, self).__init__()

        # [batch, 3, 32, 32] -> [batch, 64, 32, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        # [batch, 64, 32, 32] -> [batch, 64, 16, 16] -> [batch, 128, 8, 8] -> [batch, 256, 4, 4]
        self.blocks = nn.Sequential(
            self.block(in_channels=64, out_channels=64, num_layers=2, stride=2),
            self.block(in_channels=64, out_channels=128, num_layers=2, stride=2),
            self.block(in_channels=128, out_channels=256, num_layers=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=256 * 1 * 1, out_features=num_classes)  # 修正输入特征维度

        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def block(in_channels, out_channels, num_layers, stride=1):
        # determine whether to do down sampling
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            downsample = None

        # layers
        layers = [Block(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample)]
        for _ in range(num_layers - 1):
            # set stride = 1
            layers.append(Block(in_channels=out_channels, out_channels=out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)


class BlockWithGELU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BlockWithGELU, self).__init__()
        self.downsample = downsample

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.GELU = nn.GELU()

    def forward(self, x):
        input_copy = x
        output = self.layers(input_copy)
        if self.downsample:
            input_copy = self.downsample(input_copy)
        output += input_copy
        return self.GELU(output)


class SmallResNetWithGELU(nn.Module):
    def __init__(self, input_channels, num_classes=10):
        super(SmallResNetWithGELU, self).__init__()

        # [batch, 3, 32, 32] -> [batch, 64, 32, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.GELU()
        )

        # [batch, 64, 32, 32] -> [batch, 64, 16, 16] -> [batch, 128, 8, 8] -> [batch, 256, 4, 4]
        self.blocks = nn.Sequential(
            self.block(in_channels=64, out_channels=64, num_layers=2, stride=2),
            self.block(in_channels=64, out_channels=128, num_layers=2, stride=2),
            self.block(in_channels=128, out_channels=256, num_layers=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=256 * 1 * 1, out_features=num_classes)  # 修正输入特征维度

        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def block(in_channels, out_channels, num_layers, stride=1):
        # determine whether to do down sampling
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            downsample = None

        # layers
        layers = [BlockWithGELU(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample)]
        for _ in range(num_layers - 1):
            # set stride = 1
            layers.append(BlockWithGELU(in_channels=out_channels, out_channels=out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)


class BlockWithSwish(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BlockWithSwish, self).__init__()
        self.downsample = downsample

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.SiLU = nn.SiLU()

    def forward(self, x):
        input_copy = x
        output = self.layers(input_copy)
        if self.downsample:
            input_copy = self.downsample(input_copy)
        output += input_copy
        return self.SiLU(output)


class SmallResNetWithSwish(nn.Module):
    def __init__(self, input_channels, num_classes=10):
        super(SmallResNetWithSwish, self).__init__()

        # [batch, 3, 32, 32] -> [batch, 64, 32, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU()
        )

        # [batch, 64, 32, 32] -> [batch, 64, 16, 16] -> [batch, 128, 8, 8] -> [batch, 256, 4, 4]
        self.blocks = nn.Sequential(
            self.block(in_channels=64, out_channels=64, num_layers=2, stride=2),
            self.block(in_channels=64, out_channels=128, num_layers=2, stride=2),
            self.block(in_channels=128, out_channels=256, num_layers=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=256 * 1 * 1, out_features=num_classes)  # 修正输入特征维度

        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def block(in_channels, out_channels, num_layers, stride=1):
        # determine whether to do down sampling
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            downsample = None

        # layers
        layers = [BlockWithGELU(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample)]
        for _ in range(num_layers - 1):
            # set stride = 1
            layers.append(BlockWithGELU(in_channels=out_channels, out_channels=out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)


class ResNetLike(nn.Module):
    def __init__(self, input_channels, num_classes=10):
        super(ResNetLike, self).__init__()

        # [batch, 3, 32, 32] -> [batch, 64, 32, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        
        # [batch, 64, 32, 32] -> [batch, 64, 32, 32] -> [batch, 128, 16, 16] -> [batch, 256, 8, 8] -> [batch, 512, 4, 4]
        self.blocks = nn.Sequential(
            self.block(in_channels=64, out_channels=64, num_layers=2, stride=1),
            self.block(in_channels=64, out_channels=128, num_layers=2, stride=2),
            self.block(in_channels=128, out_channels=256, num_layers=2, stride=2),
            self.block(in_channels=256, out_channels=512, num_layers=2, stride=2),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features=512 * 1 * 1, out_features=num_classes)  # 修正输入特征维度

        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    @ staticmethod
    def block(in_channels, out_channels, num_layers, stride=1):
        # determine whether to do down sampling
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            downsample = None

        # layers
        layers = [Block(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample)]
        for _ in range(num_layers - 1):
            # set stride = 1
            layers.append(Block(in_channels=out_channels, out_channels=out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)
