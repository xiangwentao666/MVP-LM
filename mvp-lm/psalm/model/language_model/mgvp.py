import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from collections import OrderedDict

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)
    


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class GatedCrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.VISION_QUERY_CONDITION_GATE = True
        self.VISION_QUERY_NONLINEAR_GATE = True

        if self.VISION_QUERY_CONDITION_GATE:
            if self.VISION_QUERY_NONLINEAR_GATE:
                self.attn_gate = FeedForward(dim=d_model, mult=0.5, out_dim = 1)
                torch.nn.init.constant_(self.attn_gate.linear2.weight, 0)
            else:
                self.attn_gate = nn.Linear(d_model, 1, bias=False)
                torch.nn.init.constant_(self.attn_gate.weight, 0)
        else:
            self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        if self.VISION_QUERY_CONDITION_GATE:
            attn_gate = self.attn_gate(tgt2).tanh()
        else:
            attn_gate = self.attn_gate.tanh()

        tgt = tgt + attn_gate * self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt
    
    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)
    

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def FeedForward(dim, mult = 4, out_dim = None):
    inner_dim = int(dim * mult)
    if out_dim is None:
        out_dim = dim
    return nn.Sequential(
                OrderedDict([
                    ('norm', nn.LayerNorm(dim)),
                    ('linear1', nn.Linear(dim, inner_dim, bias = False)),
                    ('gelu', nn.GELU()),
                    ('linear2', nn.Linear(inner_dim, out_dim, bias = False))
                    ])
                )


class MGVPBlock(nn.Module):
    '''
    For each target category, extract one roi feature on each scale, i.e., (batch, scales, latents, dim_v), latents always = k shot.
    "latents" denotes the total length of all vison tokens at each scale.
    If the attention mask of vision v to all text t is False, return the original text embedding.
    '''
    def __init__(
        self,
        global_local_cross_layers_num = 2,
        # local_fea_dim_in = 256,
        hidden_dim = 2560,
        # dim_feedforward=4096,
        nheads = 8,
        enable_selfatten = True
    ):
        super().__init__()
        pre_norm=False,
        self.enable_selfatten = enable_selfatten
        # self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.global_local_cross_layers_num = global_local_cross_layers_num
        self.global_local_cross_layers = nn.ModuleList()
        for i in range(self.global_local_cross_layers_num):
            self.global_local_cross_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,)
            )


        # self.cross_atten = GatedCrossAttentionLayer(
        #     d_model=hidden_dim,
        #     nhead=nheads,
        #     dropout=0.0,
        #     normalize_before=pre_norm,
        # )

        # if enable_selfatten:
        #     self.self_atten = SelfAttentionLayer(
        #         d_model=hidden_dim,
        #         nhead=nheads,
        #         dropout=0.0,
        #         normalize_before=pre_norm,
        #     )

        #     self.ffn = FFNLayer(
        #         d_model=hidden_dim,
        #         dim_feedforward=dim_feedforward,
        #         dropout=0.0,
        #         normalize_before=pre_norm,
        #     )

        # learnable query p.e.

    def forward(
        self,
        x,                       # text tensor - (text_num,batch, dim)
        x_posi,
        local_vision,         # vision query tensor - (vision_num, batch, dim)
        local_vision_posi,
        memory_mask = None,   # boolean tensor indicating masks of media - (batch, vision_num, text_num)
    ):

        # global_vision = global_vision.transpose(0, 1)

        for i in range(self.global_local_cross_layers_num):
            x = self.global_local_cross_layers[i](x, local_vision, memory_mask = memory_mask,
                query_pos=x_posi, pos=local_vision_posi)


        # x = self.cross_atten(x, global_vision, memory_mask = memory_mask, query_pos=x_posi)

        # if self.enable_selfatten:
        #     x = self.self_atten(x)
        #     # apply ffn layer
        #     x = self.ffn(x)

        return x
