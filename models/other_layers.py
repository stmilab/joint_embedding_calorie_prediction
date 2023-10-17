import torch.nn as nn


class MultiTaskLayer(nn.Module):
    def __init__(self, input_size=64, hidden=128, output_size=1, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        # self.layer2 = nn.Linear(hidden, output_size)
        # self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        # x = self.act(x)
        # x = self.layer2(x)
        return x


class SingleTaskLayer(nn.Module):
    def __init__(self, input_size=128, output_size=1, dropout=0.1):
        super().__init__()
        # self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(input_size, output_size)
        # self.act = nn.ReLU()

    def forward(self, x):
        # x = self.layer1(x)
        # x = self.act(x)
        x = self.layer2(x)
        return x


class MultiModalModel(nn.Module):
    def __init__(self, models, mlp=None, use_projector=True, n_classes=1):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, xs):
        # zs = []
        # for i in range(2):
        #     print(i)
        #     zs.append(self.models[i](xs[i]))
        zs = [model(x) for model, x in zip(self.models, xs)]
        return zs


class Regressor(nn.Module):
    def __init__(self, input_size=64, hidden=128, output_size=1, dropout=0.2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        # self.layer2 = nn.Linear(hidden, output_size)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.layer3(x)
        return x
