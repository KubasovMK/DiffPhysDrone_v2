import torch
from torch import nn


class ValueNetwork(nn.Module):
    def __init__(self, dim_action=0) -> None:
        super().__init__()
        self.gru = nn.GRU(192, 192, 1)
        self.act = nn.GELU()
        self.fc =  nn.Linear(192, 1)
        self.proj = nn.Linear(dim_action + 192, 192)
        self.proj.weight.data[:, :dim_action] *= ((dim_action + 192) / dim_action / 2) ** 0.5
        self.proj.weight.data[:, dim_action:] *= ((dim_action + 192) / 192 / 2) ** 0.5

    def forward(self, x):
        x = self.proj(x).relu_()
        return self.fc(self.act(self.gru(x)[0]))


class Critic(nn.Module):
    def __init__(self, dim_obs) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.observation_fc = nn.Linear(dim_obs, 192)

        self.gru = nn.GRU(192, 192, 1)
        self.action_fc = nn.Linear(192, 1, bias=False)
        self.activation = nn.GELU()

    def forward(self, x, v):
        T, B, C, H, W = x.shape
        img_feat = self.stem(x.flatten(0, 1))
        img_feat = img_feat.reshape(T, B, -1)
        x = self.activation(img_feat + self.observation_fc(v))
        return self.action_fc(self.activation(self.gru(x)[0]))


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.dim_obs = dim_obs
        self.observation_fc = nn.Linear(dim_obs, 192)

        self.gru = nn.GRUCell(192, 192)
        self.action_fc = nn.Linear(192, dim_action, bias=False)
        self.activation = nn.LeakyReLU(0.05)

        # balance feature weight
        self.observation_fc.weight.data.mul_(0.5)
        # self.action_fc.weight.data.mul_(0.01)

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.activation(img_feat + self.observation_fc(v))
        hx = self.gru(x, hx)
        action = self.action_fc(self.activation(hx))
        return action, hx

    def batch_forward(self, x: torch.Tensor, v, hx=None):
        T, B, C, H, W = x.shape
        img_feat = self.stem(x.flatten(0, 1))
        img_feat = img_feat.reshape(T, B, -1)
        x = self.activation(img_feat + self.observation_fc(v))
        hx_out = []
        for x in x.unbind(0):
            hx = self.gru(x, hx)
            hx_out.append(hx)
        hx = torch.stack(hx_out)
        action = self.action_fc(self.activation(hx))
        return action


if __name__ == '__main__':
    Model()
