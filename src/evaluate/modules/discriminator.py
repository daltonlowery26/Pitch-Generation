from torch import nn

from src.utils.modeling.opti import tag


class Residual(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    def __init__(self, inp, dim, blocks):
        super().__init__()
        self.inp = inp
        self.dim = dim
        self.blocks = blocks

        struct = [nn.Sequential(nn.RMSNorm(inp), nn.Linear(inp, dim * 4), nn.SiLU())]

        for _ in range(blocks):
            struct.append(
                Residual(
                    nn.Sequential(
                        nn.RMSNorm(dim * 4), nn.Linear(dim * 4, dim * 4), nn.SiLU()
                    )
                )
            )

        struct.append(nn.Linear(dim * 4, 1))
        self.layers = nn.Sequential(*struct)

        tag(self.layers[-1], no_muon=True)

    def forward(self, embed):
        return self.layers(embed)
