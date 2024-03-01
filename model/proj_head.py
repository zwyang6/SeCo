import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Borrow from https://github.com/facebookresearch/dino
"""

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=768):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1,)
        return x

class DINOHeadSep(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3,
    ):
        super().__init__()
        if nlayers == 1:
            self.mlp1 = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp1 = nn.Sequential(*layers)
        self.mlp1.apply(self._init_weights)

        self.last_layer1 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer1.weight_g.data.fill_(1)
        self.last_layer1.weight_g.requires_grad = False

        if nlayers == 1:
            self.mlp2 = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp2 = nn.Sequential(*layers)
        self.mlp2.apply(self._init_weights)

        self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer2.weight_g.data.fill_(1)
        self.last_layer2.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.mlp1(x[:, :1])
        x1 = F.normalize(x1, dim=-1, p=2)
        x1 = self.last_layer1(x1)

        x2 = self.mlp2(x[:, 1:])
        x2 = F.normalize(x2, dim=-1, p=2)
        x2 = self.last_layer2(x2)

        x = torch.cat([x1, x2], dim=1)
        return x
