from torch.nn import Module, ModuleList, Linear, BatchNorm1d, Tanh, Sigmoid


class ResBlock(Module):
    def __init__(self, in_channel: int, mid_channel: int) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.nn1 = Linear(in_channel, mid_channel)
        self.norm1 = BatchNorm1d(mid_channel)
        self.nn2 = Linear(mid_channel, mid_channel)
        self.norm2 = BatchNorm1d(mid_channel)
        self.nn3 = Linear(mid_channel, in_channel)
        self.norm3 = BatchNorm1d(in_channel)
        self.activation = Tanh()

    def forward(self, x):
        # 將原始值分出
        res = x

        # 計算殘差層
        x = self.nn1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # 計算殘差層
        x = self.nn2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # 計算殘差層
        x = self.nn3(x)
        x = self.norm3(x)
        x = self.activation(x)

        # 殘差加入
        x = res + x
        return x


class GatingModel(Module):
    def __init__(self):
        super(GatingModel, self).__init__()
        blocks = [ResBlock(1024, 1024) for _ in range(5)]
        self.res_blocks = ModuleList(blocks)
        self.nn1 = Linear(1024, 512)
        self.nn2 = Linear(512, 128)
        self.nn3 = Linear(128, 16)
        self.nn4 = Linear(16, 3)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        for res in self.res_blocks:
            x = res(x)

        x = self.nn1(x)
        self.tanh(x)

        x = self.nn2(x)
        self.tanh(x)

        x = self.nn3(x)
        self.tanh(x)

        x = self.nn4(x)
        x = self.sigmoid(x)

        return x


class NaiveGatingModel(Module):
    def __init__(self):
        super().__init__()
        self.nn1 = Linear(1024, 256)
        self.nn2 = Linear(256, 64)
        self.nn3 = Linear(64, 3)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.nn1(x)
        self.tanh(x)

        x = self.nn2(x)
        self.tanh(x)

        x = self.nn3(x)
        x = self.sigmoid(x)

        return x
