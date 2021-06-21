from collections import OrderedDict
import math

import torch
from torch import distributed as dist
from torch import nn

from vimpac.modeling_utils import PAD_TOKEN_ID, LayerNorm, QuickGELU
from vimpac.positional_embedding import PositionalEmbedding


class BertSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout_p=0., args=None):
        super().__init__()
        assert output_dim % num_heads == 0

        self.num_attention_heads = num_heads
        self.attention_head_size = output_dim // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout_p)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def extra_repr(self) -> str:
        return f"Heads: {self.num_attention_heads}"


class BertSelfOutput(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.LayerNorm = nn.LayerNorm(output_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SingleAttention(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_head: int, dropout: float = 0.0, pattern="T", args=None):
        super().__init__()
        self.attention = BertSelfAttention(input_dim, d_model, n_head, dropout, args=args)
        self.pattern = pattern

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """

        :param x: B, L, H, W, D
        :param mask: B, L, H, W
        :return: B, L, H, W, D
        """
        cls, x = x
        batch_size = x.shape[0]
        hidden_dim = x.shape[-1]

        if self.pattern == "H":             # 0  1  2  3  4
            transpose = (2, 3)              # B, L, H, W, D -- B, L, W, H, D
        elif self.pattern == "T":           # 0  1  2  3  4
            transpose = (1, 3)              # B, L, H, W, D -- B, H, W, L, D
        elif self.pattern == "W":
            transpose = None
        else:
            assert False

        # Here, we take H, major as an example
        if transpose is not None:
            x = x.transpose(*transpose)                 # B, L, H, W, D --> B, L, W, H, D
            if mask is not None:
                # mask = mask.permute(0, *permutation)
                mask = mask.transpose(*transpose)       # B, L, H, W    --> B, L, W, H

        # Pad the CLS token
        x = torch.cat([cls.reshape(batch_size, 1, 1, 1, -1) + torch.zeros(*x.shape[:-2], 1, hidden_dim, dtype=x.dtype, device=x.device),
                       x], dim=3)  # (B, L, W, 1, Dim) + (B, L, W, H, Dim) -->  B, L, W, 1+H, Dim
        if mask is not None:
            mask = torch.cat([torch.zeros(*x.shape[:-2], 1, dtype=mask.dtype, device=mask.device),
                              mask], dim=3)

        batch_shape = x.shape[:-2]      # B, L, W,
        seq_len = x.shape[-2]           # 1 + H
        x = self.attention(
            x.reshape(-1, seq_len, hidden_dim),
            mask.reshape(-1, seq_len) if mask is not None else mask
        )

        x = x.reshape(*batch_shape, seq_len, -1)        # B, L, W, 1+H, D

        # split to cls and x
        cls = x[..., 0, :].mean(1).mean(1)
        x = x[..., 1:, :]

        if transpose is not None:
            x = x.transpose(*transpose)

        return cls, x

    def extra_repr(self) -> str:
        PATTERN2NAME = {
            "T": "Temporal",
            "W": "Width",
            "H": "Height",
        }
        return f"Efficient Attention over {PATTERN2NAME[self.pattern]}"


class MultiPatternAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, patterns: str, pre_activation=False, dropout=0.0, args=None):
        super(MultiPatternAttention, self).__init__()
        self.pre_activation = pre_activation

        patterns = patterns.split("|")
        assert d_model % len(patterns) == 0
        assert n_head % len(patterns) == 0

        d_attention = d_model // 2 if args is not None and args.halve_att else d_model

        self.attentions = nn.ModuleList(
            SingleAttention(d_model, d_attention // len(patterns), n_head // len(patterns), dropout=dropout, pattern=pattern, args=args)
            for pattern in patterns
        )

        if not self.pre_activation:
            # still keep the name of BertSelfOutput to support model loading.
            self.output = BertSelfOutput(d_attention, d_model, dropout=dropout)
        else:
            self.dense = nn.Linear(d_model, d_model)
            self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
            self.dropout = nn.Dropout(dropout)

    def preactivation_output(self, hiddens, x):
        return self.dropout(self.dense(hiddens)) + x

    def forward(self, x, mask=None):
        cls, x = x

        if self.pre_activation:
            cls = self.LayerNorm(cls)
            x = self.LayerNorm(x)

        att_cls, att_ctx = zip(*[attention((cls, x), mask) for attention in self.attentions])
        if len(att_cls) > 1:
            att_cls = torch.cat(att_cls, -1)
            att_ctx = torch.cat(att_ctx, -1)
        else:
            att_cls = att_cls[0]
            att_ctx = att_ctx[0]

        if not self.pre_activation:
            cls_output = self.output(att_cls, cls)
            x_output = self.output(att_ctx, x)
        else:
            cls_output = self.preactivation_output(att_cls, cls)
            x_output = self.preactivation_output(att_ctx, x)

        return cls_output, x_output


class ResidualAttentionBlock(nn.Module):
    """
    Clarification from TimesFormer Author (Gedas Bertasius):
    Before applying each attention, you can replicate the CLS token N times,
    where N is the number of spatial/temporal tokens over which the attention will be computed.
    After the attention is applied, you can average these CLS tokens to obtain a single token.
    Let me know if this makes sense.
    """
    def __init__(self, d_model: int, n_head: int, pre_activation: bool = False,
                 dropout: float = 0.1, layout="T,H,W", grad_checkpoint: bool = False, args=None):
        super().__init__()
        self.pre_activation = pre_activation
        self.grad_checkpoint = grad_checkpoint

        self.attention_blocks = nn.ModuleList([
            MultiPatternAttention(d_model, n_head, dropout=dropout, pre_activation=pre_activation, patterns=patterns, args=args)
            for patterns in layout.split(",")
        ])

        d_mlp = d_model * 2 if args is not None and args.halve_mlp else d_model * 4

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_mlp, d_model)),
            ("dropout", nn.Dropout(dropout)),
        ]))
        self.ln = LayerNorm(d_model, eps=1e-12)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        :param x: (1 + L), (1 + H x W), B, Dim
        :param mask: (1 + L), (1 + H x W), B
        :return: (1 + L), (1 + H x W)
        """
        for attention_block in self.attention_blocks:
            x = attention_block(x, mask)

        cls, x = x
        if self.pre_activation:
            cls = cls + self.mlp(self.ln(cls))
            x = x + self.mlp(self.ln(x))
        else:
            cls = self.ln(cls + self.mlp(cls))
            x = self.ln(x + self.mlp(x))

        return cls, x


class Transformer(nn.Module):
    def __init__(self, hid_dim: int, layers: int, heads: int,
                 dropout: float = 0.1, layout: str = "T,H,W", pre_activation: bool = False,
                 grad_checkpoint=False, args=None):
        super().__init__()
        self.width = hid_dim
        self.layers = layers
        self.grad_checkpoint = grad_checkpoint
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(hid_dim, heads, dropout=dropout, layout=layout, pre_activation=pre_activation,
                                    grad_checkpoint=grad_checkpoint, args=args)
             for _ in range(layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for res_block in self.resblocks:
            if self.grad_checkpoint:
                def create_custom_forward(module):
                    def custom_forward(cls, hiddens, mask):
                        return module((cls, hiddens), mask)

                    return custom_forward

                cls, hiddens = x
                cls, hiddens = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(res_block),
                    cls,
                    hiddens,
                    mask,
                )
                x = (cls, hiddens)
            else:
                x = res_block(x, mask)
        return x


class TransformerLayout(nn.Module):
    def __init__(self,
                 height: int = 32,
                 width: int = 32,
                 length: int = 10,
                 vocab_size: int = 8196,
                 hid_dim: int = 512,
                 layers: int = 12,
                 heads: int = 8,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 pre_activation: bool = False,
                 output_mode: str = "cls",
                 pos_emb_args: dict = {},
                 layout: str = "T,H,W",
                 grad_checkpoint: bool = False,
                 args = None,
                 ):
        """

        :param height: the input spatial height
        :param width: the input spatial width; Note: for imagenet, we use width to indicate hid_dim following OA's CLIP
        :param length: the input temporal length
        :param vocab_size: the number of tokens
        :param hid_dim: hidden dimension
        :param layers: number of layers
        :param heads: head of the attention layers
        :param use_cls_token: use the additional CLS token.
        :param pre_activation: ViT style of pre activation
        """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.pre_activation = pre_activation
        self.output_mode = output_mode
        self.initializer_range = 0.02
        self.length = length
        self.height = height
        self.width = width
        self.grad_checkpoint = grad_checkpoint

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.drop = nn.Dropout(dropout)
        if use_cls_token:
            # This initialization is the same as BERT
            self.cls_param = nn.Parameter(self.initializer_range * torch.randn(hid_dim))
            if dist.get_rank() == 0:
                print("Add [CLS] token inside the decoder.")

        self.positional_embedding = PositionalEmbedding(
            height, width, length, hid_dim,
            use_cls_token=use_cls_token, initializer_range=self.initializer_range,
            **pos_emb_args,
        )

        if not pre_activation:
            # define it before transformer for supporting loading old models.
            self.ln_pre = LayerNorm(hid_dim, eps=1e-12)

        self.transformer = Transformer(
            hid_dim, layers, heads,
            layout=layout, dropout=dropout, pre_activation=pre_activation,
            grad_checkpoint=grad_checkpoint,
            args=args,
        )

        if pre_activation:
            self.ln_post = LayerNorm(hid_dim, eps=1e-12)

        # Weight initialization
        self.apply(self._bert_init_weights)
        if pre_activation:
            self._preactivation_init_weights()      # Init the

    def forward(self,
                x: torch.Tensor,
                output_mode: str = None,
                ):
        """
        :param x: B, L, H, W (int)
        :param output_mode: overwrite the default output_mode.
        :return:  B, Dim (float)
        """
        b, l, h, w = x.shape

        mask = (x == PAD_TOKEN_ID)      # B, L, H, W
        if (~mask).all():
            mask = None

        x = self.embedding(x)  # B, L, H, W --> B, L, H, W, Dim
        cls = self.cls_param.repeat(b, 1)

        # Reshape the positional embedding
        pos_emb = self.positional_embedding().to(x.dtype)   # 1 + L x H x W, Dim
        assert pos_emb.shape[-1] == x.shape[-1]

        cls_pos_emb = pos_emb[0]                            # Dim,
        lhw_pos_emb = pos_emb[1:].reshape(self.length, self.height, self.width, -1)      # L, H, W, Dim
        assert cls_pos_emb.shape[-1] == lhw_pos_emb.shape[-1] == x.shape[-1] == cls.shape[-1], \
            (cls_pos_emb.shape[-1], lhw_pos_emb.shape[-1], x.shape[-1], cls.shape[-1])

        # Add token emb and pos emb, then LayerNorm
        if self.pre_activation:
            cls = self.drop(cls + cls_pos_emb)
            x = self.drop(x + lhw_pos_emb)
        else:
            cls = self.drop(self.ln_pre(cls + cls_pos_emb))
            x = self.drop(self.ln_pre(x + lhw_pos_emb))

        cls, x = self.transformer((cls, x), mask)       # 1+L, 1+HxW, B, Dim --> 1+L, 1+HxW, B, Dim

        if self.pre_activation:
            # For ViT style, we need an additional LayerNorm to normalize the residual path
            # However, for the default BERT-style, the last operator is LN thus no additional LN is needed.
            cls = self.ln_post(cls)
            x = self.ln_post(x)

        if output_mode is None:
            output_mode = self.output_mode
        if output_mode == "cls":
            return cls
        elif output_mode == "hiddens":
            return x
        elif output_mode == "both":
            return cls, x
        elif output_mode == "pooling":
            return x.mean(1).mean(1).mean(1)        # B, L, H, M, D --> B, D

    def _bert_init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=self.initializer_range)
            module.out_proj.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # if hasattr(module"in_proj_bias"):
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            # else:
            #     assert False
            # if hasattr(module, "out_proj.bias"):
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
            # else:
            #     assert False
        elif isinstance(module, nn.LayerNorm) or isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _preactivation_init_weights(self):
        num_residual_connections = 0
        for resblock in self.transformer.resblocks:
            num_attention_blocks = len(resblock.attention_blocks)
            num_residual_connections += (num_attention_blocks + 1)      # num_layers = num_attention + MLP

        normalizer = 1 / math.sqrt(num_residual_connections)
        if dist.get_rank() == 0:
            print("Pre-activation initialization:")
            print(f"\tscale the dense proj layer by {self.initializer_range} * {normalizer} (1 / sqrt of {num_residual_connections})")
        cnt = 0
        for resblock in self.transformer.resblocks:
            resblock.mlp.c_proj.weight.data.normal_(mean=0.0, std=self.initializer_range * normalizer)
            cnt += 1
            for attn_block in resblock.attention_blocks:
                attn_block.dense.weight.data.normal_(mean=0.0, std=self.initializer_range * normalizer)
                cnt += 1
        assert cnt == num_residual_connections



class BertLMPredictionHead(nn.Module):
    def __init__(self,
                 hid_dim: int,
                 input_dim: int = None,
                 output_labels: int = 8192,
                 tied_embedding: nn.Embedding = None,
                 output_bias: bool = True,
                 ):
        super().__init__()
        self.initializer_range = 0.02

        input_dim = hid_dim if input_dim is None else input_dim

        self.transform = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            QuickGELU(),
            LayerNorm(hid_dim, eps=1e-12),
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hid_dim, output_labels, bias=False)

        if output_bias:
            self.bias = nn.Parameter(torch.zeros(output_labels))

            # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
            self.decoder.bias = self.bias

        self.apply(self._bert_init_weights)

        # Remember to put this tie after thee weight initialization!
        if tied_embedding is not None:
            self.decoder.weight = tied_embedding.weight

    def forward(self, hidden_states):
        """
        :param hidden_states: [batch_size, length, hid_dim]
        :return:              [batch_size, length, output_labels]
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _bert_init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CLSMLPHead(nn.Module):
    def __init__(self, num_layers=2, input_dim=768, hid_dim=4096, output_dim=256,
                 norm_type='bn', activation="relu", bn_after_proj=False):
        """
        A projection head over the CLS token.
        This implementation closely follow the original implementation in https://github.com/google-research/simclr

        :param num_layers: number of layers in the projection head
        :param input_dim: output dim of the backbone model
        :param hid_dim: hidden dimension of the MLP
        :param output_dim: output dim of this projection head (for future contrastive learning)
        :param norm_type: the default is BN (batch norm) in simclr, but we support layer norm here.
                          according to MoCoV3, BN is better (but it is compared with no norm layers?)
        :param activation: non linear in MLP
        :param bn_after_proj: whether adding an additional BN after the projection head. It is used in SimCLR.
        """
        super(CLSMLPHead, self).__init__()

        if activation == "relu":
            ACT = nn.ReLU
        elif activation == "gelu":
            ACT = QuickGELU
        else:
            assert False

        layers = []
        for layer_id in range(num_layers):
            fan_in = input_dim if layer_id == 0 else hid_dim
            fan_out = output_dim if layer_id == (num_layers - 1) else hid_dim

            if layer_id == num_layers - 1:      # The last layer
                layers.append(
                    (f"linear_{layer_id}", nn.Linear(fan_in, fan_out, bias=False))
                )

                if norm_type == "bn" and bn_after_proj:
                    # SimCLR adds one BatchNorm after the final linear layer.
                    #   (it will be further L2-normalized inside the NCE loss (nce_support.py) )
                    layers.append(
                        (f"bn_{layer_id}", nn.SyncBatchNorm(fan_out))
                    )
            else:
                if norm_type == "bn":
                    layers.extend([
                        (f"linear_{layer_id}", nn.Linear(fan_in, fan_out)),
                        (f"bn_{layer_id}", nn.SyncBatchNorm(fan_out)),
                        (f"act_{layer_id}", ACT()),
                    ])
                elif norm_type == "ln":
                    layers.extend([
                        (f"linear_{layer_id}", nn.Linear(fan_in, fan_out)),
                        (f"act_{layer_id}", ACT()),
                        (f"ln_{layer_id}", LayerNorm(fan_out, eps=1e-12)),
                    ])

        self.mlp = nn.Sequential(OrderedDict(layers))

        self.initializer_range = 0.02
        self.apply(self._bert_init_weights)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x

    def _bert_init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ModelWithHead(nn.Module):
    def __init__(self, model, mlm_head, cls_head=None):
        super(ModelWithHead, self).__init__()
        self.backbone = model
        self.mlm_head = mlm_head
        self.cls_head = cls_head
    
    def forward(self, x: torch.Tensor, *args, **kwargs):
        cls, x = self.backbone(x, *args, **kwargs)
        x = self.mlm_head(x)

        if self.cls_head is not None:
            cls_output = self.cls_head(cls)
            return cls_output, x
        else:
            return x


