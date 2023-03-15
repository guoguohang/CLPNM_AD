import torch.nn as nn


class TabEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, channel_input=16, sign_size=8, cha_hidden=32):
        super(TabEncoder, self).__init__()
        self.channel_input = channel_input
        self.sign_size = sign_size
        self.trans = nn.Sequential(
            nn.Linear(input_dim, channel_input * sign_size)
        )
        self.encoder = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel_input, cha_hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(cha_hidden),
            nn.LeakyReLU(0.2),
            nn.Conv1d(cha_hidden, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(output_dim, output_dim, kernel_size=1, bias=True),
            nn.BatchNorm1d(output_dim),
            nn.Flatten(),
            nn.LeakyReLU(0.2),
            nn.Linear(output_dim * sign_size, output_dim)
        )
        self.proj_head = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        x = self.trans(x)
        x = x.reshape(x.shape[0], self.channel_input, self.sign_size)
        embedding = self.encoder(x)
        feature = self.proj_head(embedding)
        return embedding, feature
