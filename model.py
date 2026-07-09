import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(
        self,
        dim_obs=9,
        dim_action=4,
        traj_points=0,
        traj_dim=6,
    ) -> None:
        super().__init__()

        self.traj_points = traj_points
        self.traj_dim = traj_dim
        self.use_traj = traj_points > 0

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),   # 32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),  # 64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )

        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        if self.use_traj:
            self.traj_proj = nn.Sequential(
                nn.Linear(traj_points * traj_dim, 128),
                nn.LeakyReLU(0.05),
                nn.Linear(128, 192, bias=False),
            )
            self.traj_proj[-1].weight.data.mul_(0.1)
        else:
            self.traj_proj = None

        self.gru = nn.GRUCell(192, 192)

        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)

        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None, traj=None):
        img_feat = self.stem(x)
        feat = img_feat + self.v_proj(v)

        if self.use_traj:
            if traj is None:
                raise ValueError("Model was created with traj_points > 0, but traj=None was passed.")

            traj = traj.reshape(traj.shape[0], self.traj_points * self.traj_dim)
            feat = feat + self.traj_proj(traj)

        x = self.act(feat)
        hx = self.gru(x, hx)

        act = self.fc(self.act(hx))
        return act, None, hx



if __name__ == '__main__':
    Model()
