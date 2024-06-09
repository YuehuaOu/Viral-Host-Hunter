import torch
import torch.nn as nn
from config import config


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Input_layer(nn.Module):
    def __init__(self, patch_size, embed_size, device, dropout):
        super(Input_layer, self).__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.device = device
        self.dropout = dropout
        self.patch_embedding = nn.Linear(patch_size, embed_size)
        self.position_embedding = nn.Embedding(40, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length, embed_size = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.patch_embedding(x) + self.position_embedding(positions))
        )
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            device,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.layers = TransformerBlock(
            embed_size,
            heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
        )

    def forward(self, x, mask):
        out = self.layers(x, x, x, mask)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            patch_size,
            embed_size=128,
            num_layers=1,
            forward_expansion=4,
            heads=8,
            dropout=0.4,
            device="cpu",
            max_length=100,
    ):
        super(Transformer, self).__init__()

        self.input = Input_layer(patch_size, embed_size, device, dropout)

        self.encoders = nn.ModuleList([Encoder(embed_size, heads, device, forward_expansion, dropout)
                                       for _ in range(num_layers)])

        self.device = device
        self.max_length = max_length

    def make_src_mask(self, src):
        src_mask = torch.ones(self.max_length).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src):
        src_mask = self.make_src_mask(src)
        enc_src = self.input(src)

        for encoder in self.encoders:
            enc_src = encoder(enc_src, src_mask)
        enc_src = enc_src.reshape(enc_src.shape[0], -1)
        return enc_src


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=20, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(20)

        self.conv3 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(10)

        self.conv5 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=2, stride=1)
        self.bn5 = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=2, stride=1)
        self.bn6 = nn.BatchNorm2d(10)
        self.transformer = Transformer(patch_size=(2 ** config.k - 2) ** 2, embed_size=128, num_layers=1,
                                       device=config.device, forward_expansion=4)

    def forward(self, x, p_embed):
        out = self.relu(self.bn1(self.conv1(x)))
        out1 = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(x)))
        out2 = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(x)))
        out3 = self.relu(self.bn6(self.conv6(out)))
        out = torch.cat((out1, out2, out3), dim=1)
        out = nn.Dropout(p=0.2)(out)
        new_shape = (out.shape[0], out.shape[1], -1)
        out = out.view(*new_shape)
        out = self.transformer(out)
        out = torch.cat((out, p_embed), dim=1)
        return out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(in_features=133 + 1024 + 1280*4, out_features=2048)
        self.bn_l1 = nn.BatchNorm1d(2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=1024)
        self.bn_l2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=config.num_class)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dropout(self.relu(self.bn_l1(self.linear1(x))))
        out = self.dropout(self.relu(self.bn_l2(self.linear2(out))))
        out = self.linear3(out)
        return out


class DnaPathNetworks(nn.Module):
    def __init__(self):
        super(DnaPathNetworks, self).__init__()
        self.feature = FeatureExtractor()
        self.classifier = Classifier()

    def forward(self, x, embed):
        out_feature = self.feature(x, embed)
        out = self.classifier(out_feature)
        return out
