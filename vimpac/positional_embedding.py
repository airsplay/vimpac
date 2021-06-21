import math

import torch
from torch import nn, distributed as dist


OPTION2ARGS = {
    "default": {},
    "temporal_separation": {
        "temporal_separation": True,
    },
    "hw_separation": {
        "temporal_separation": True,
        "hw_separation": True,
    },
    "pre_defined": {
        "temporal_separation": True,
        "hw_separation": True,
        "pre_defined": True,
    },
    "frozen": {
        "temporal_separation": True,
        "hw_separation": True,
        "pre_defined": True,
        "frozen": True,
    },
}


def option2args(option: str = "default"):
    return OPTION2ARGS[option]


class PositionalEmbedding(nn.Module):
    def __init__(self, height, width, length, hid_dim, use_cls_token=False, initializer_range=0.02,
                 temporal_separation=False, hw_separation=False, pre_defined=False, frozen=False, **kwargs):
        """

        :param temporal_separation: If True, use temporal_pos_emb + spatial_pos_emb
        :param hw_separation: height_width_separation; If True, use temporal_pos_emb + height_pos_emb + width_pos_emb
        :param pre_defined: whether use the sin/cos feature
        :param frozen: whether freeze the position embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.temporal_separation = temporal_separation
        self.hw_separation = hw_separation
        self.hid_dim = hid_dim
        self.length = length
        self.height = height
        self.width = width
        self.use_cls_token = use_cls_token

        cls_len = 1 if use_cls_token else 0
        input_size = height * width * length + cls_len
        if not temporal_separation and not hw_separation:
            self.positional_embedding = nn.Parameter(initializer_range * torch.randn(input_size, hid_dim))

        if dist.get_rank() == 0:
            print(f"Positional Embedding: Use a length of {input_size}.")

        if temporal_separation and not hw_separation:
            if use_cls_token:
                self.cls_pos_embedding = nn.Parameter(initializer_range * torch.randn(1, hid_dim))
            initializer_range = initializer_range / math.sqrt(2)
            self.temporal_pos_embedding = nn.Parameter(initializer_range * torch.randn(length, hid_dim))
            self.spatial_pos_embedding = nn.Parameter(initializer_range * torch.randn(height * width, hid_dim))

        if hw_separation:
            assert temporal_separation
            if use_cls_token:
                self.cls_pos_embedding = nn.Parameter(initializer_range * torch.randn(1, hid_dim))
            initializer_range = initializer_range / math.sqrt(3)
            self.temporal_pos_embedding = nn.Parameter(initializer_range * torch.randn(length, hid_dim))
            self.height_pos_embedding = nn.Parameter(initializer_range * torch.randn(height, hid_dim))
            self.width_pos_embedding = nn.Parameter(initializer_range * torch.randn(width, hid_dim))

        if pre_defined:
            assert hw_separation
            raise NotImplementedError

        if frozen:
            assert hw_separation and pre_defined
            raise NotImplementedError

    def forward(self):
        if self.temporal_separation:
            if self.hw_separation:
                # temporal + height + width
                pos_embedding = torch.reshape(
                    input=(self.temporal_pos_embedding.reshape(self.length, 1, 1, -1) +
                           self.height_pos_embedding.reshape(1, self.height, 1, -1) +
                           self.width_pos_embedding.reshape(1, 1, self.width, -1)
                           ),
                    shape=(self.length * self.height * self.width, -1)
                )
                if self.use_cls_token:
                    pos_embedding = torch.cat([self.cls_pos_embedding, pos_embedding])
                return pos_embedding
            else:
                # temporal + spatial
                pos_embedding = torch.reshape(
                    input=(self.temporal_pos_embedding.reshape(self.length, 1, -1) +
                           self.spatial_pos_embedding.reshape(1, self.height * self.width, -1)
                           ),
                    shape=(self.length * self.height * self.width, -1)
                )
                pos_embedding = torch.cat([self.cls_pos_embedding, pos_embedding])
                return pos_embedding
        else:
            return self.positional_embedding

    def extra_repr(self):
        if self.temporal_separation:
            if self.hw_separation:
                return f"LHW separation, L {self.temporal_pos_embedding.shape}, " \
                       f"H {self.height_pos_embedding.shape}, W {self.width_pos_embedding.shape}"
            else:
                return f"LS separation, L {self.temporal_pos_embedding.shape}, " \
                       f"S {self.spatial_pos_embedding.shape}"
        else:
            return f"{self.positional_embedding.shape}"