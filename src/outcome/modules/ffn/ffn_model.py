from torch import nn


class Residual(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def __forward(self, x):
        return x + self.block(x)


class Pitch_Predict(nn.Module):
    def __init__(self, inp, out, dim, blocks):
        super().__init__()
        # model vars
        self.inp = inp
        self.out = out
        self.dim = dim
        self.blocks = blocks

        # standard input
        struct = [nn.Sequential(nn.RMSNorm(inp), nn.Linear(dim, dim * 4), nn.SiLU())]

        # allow for specification of feed forward
        for _ in range(blocks):
            struct.append(
                Residual(
                    nn.Sequential(
                        nn.RMSNorm(dim * 4), nn.Linear(dim * 4, dim * 4), nn.SiLU()
                    )
                )
            )

        struct.append(nn.Linear(dim * 4, out))
        self.layers = nn.Sequential(*struct)

    def __forward__(self, embed):
        return self.layers(embed)
