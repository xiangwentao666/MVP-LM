# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .position_encoding import PositionEmbeddingSine
from .utils import gen_encoder_output_proposals, inverse_sigmoid
from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
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

        self.activation = _get_activation_fn(activation)
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


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
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


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def get_bounding_boxes(masks_tensor):
    """
    Returns:
        Boxes: tight bounding boxes around bitmasks.
        If a mask is empty, it's bounding box will be all zero.
    """

    boxes = torch.zeros(masks_tensor.shape[0], 4, dtype=torch.float32)
    x_any = torch.any(masks_tensor>0, dim=1) # unsigmod box tensor
    y_any = torch.any(masks_tensor>0, dim=2)
    for idx in range(masks_tensor.shape[0]):
        x = torch.where(x_any[idx, :])[0]
        y = torch.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = torch.as_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
            )
    return boxes

class MultiScaleMaskedTransformerDecoderForOPTPreTrain(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            total_num_feature_levels: int = 4,
            seg_norm=False,
            seg_concat=True,
            seg_proj=True,
            seg_fuse_score=False,
            initial_pred:bool = True,
            dn='seg',
            noise_scale=0.4,
            dn_num=100,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            num_queries_stuff=100,
    ):
        nn.Module.__init__(self)
        # # positional encoding
        # N_steps = hidden_dim // 2
        # self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        # self.transformer_self_attention_layers = nn.ModuleList()
        # self.transformer_cross_attention_layers = nn.ModuleList()
        # self.transformer_ffn_layers = nn.ModuleList()

        # for _ in range(self.num_layers):
        #     self.transformer_self_attention_layers.append(
        #         SelfAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=nheads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        #     self.transformer_cross_attention_layers.append(
        #         CrossAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=nheads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        #     self.transformer_ffn_layers.append(
        #         FFNLayer(
        #             d_model=hidden_dim,
        #             dim_feedforward=dim_feedforward,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        self.seg_norm = seg_norm
        self.seg_concat = seg_concat
        self.seg_proj = seg_proj
        self.seg_fuse_score = seg_fuse_score
        if self.seg_norm:
            print('add seg norm for [SEG]')
            self.seg_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.class_name_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.SEG_norm = nn.LayerNorm(hidden_dim)
            self.class_name_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.num_queries_stuff=num_queries_stuff
        # self.query_feat = nn.Embedding(num_queries_stuff, hidden_dim)
        # self.query_embed = nn.Embedding(num_queries_stuff, 4)

        self.SEG_query_embed = nn.Embedding(num_queries, 4)
        self.SEG_query_feat = nn.Embedding(num_queries, hidden_dim)

        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = total_num_feature_levels
        # self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.SEG_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.CLASS_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.REGION_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        # self.logit_scale = nn.Parameter(torch.ones([])) # class name sim scale
        self.dn = dn
        self.noise_scale=noise_scale
        self.dn_num=dn_num

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )
        
        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

        self.initial_pred=initial_pred
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    

    def prepare_for_dn(self, targets, class_name_embedding, batch_size,task="other"):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num) > 0:
                scalar = scalar // (int(max(known_num)))
            else:
                scalar = 0
            if task=="cls":
                scalar=1
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            # use languge as denosing content queries.
            # if task == 'det':
            #     labels = labels  # o365 start from 133 class
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            # input_label_embed = torch.gather(self.lang_encoder.default_text_embeddings, 0,
            #                              m[:, None].repeat(1, self.dim_proj)) @ self.lang_mapper
            m = m + known_bid*class_name_embedding.shape[1]
            # same class name embedding within openseed batch; different embedding within psalm batch d
            input_label_embed = torch.gather(class_name_embedding.flatten(0,1), 0,
                                              m[:, None].repeat(1, class_name_embedding.shape[-1])) 

            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = input_label_embed.new_zeros(pad_size, self.hidden_dim)
            padding_bbox = input_bbox_embed.new_zeros(pad_size, 4)


            input_query_label = padding_label.repeat(batch_size, 1, 1)
            input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = input_label_embed.new_tensor([])
            if len(known_num):
                map_known_indice = torch.cat(
                    [input_label_embed.new_tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            if 'panoptic' in task:
                tgt_size = pad_size + self.num_queries+self.num_queries_stuff
            else:
                tgt_size = pad_size + self.num_queries

            attn_mask = input_label_embed.new_ones(tgt_size, tgt_size) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            input_query_label = None
            input_query_bbox = None
            attn_mask = None
            mask_dict = None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict
    
    def dn_post_process(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        output_known_mask = None
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_SEG_logits': None,
               'pred_class_name_logits': output_known_class[-1], 
               'pred_region_logits': None,
               'pred_boxes': output_known_coord[-1],
               'pred_masks': None if output_known_mask is None else output_known_mask[-1]}

        
        out['aux_outputs'] = self._set_aux_loss([None]*len(output_known_class), output_known_class,
                                                output_known_mask, [None]*len(output_known_class), output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask


    def forward(self, x, mask_features, proposal=None, seg_query=None, SEG_embedding=None, class_name_embedding=None, region_embedding_list=None, targets=None, tasks='pano'):
        # if self.seg_concat:
        #     return self.forward_concat(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding, region_embedding_list)
        # else:
        return self.forward_woconcat(x, mask_features, proposal, seg_query, SEG_embedding, class_name_embedding, region_embedding_list, targets, tasks)

    # def forward_concat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None,
    #                    class_name_embedding=None, region_embedding_list=None):
    #     # x is a list of multi-scale feature
    #     assert len(x) == self.num_feature_levels
    #     src = []
    #     pos = []
    #     size_list = []

    #     # disable mask, it does not affect performance
    #     del mask

    #     for i in range(self.num_feature_levels):
    #         size_list.append(x[i].shape[-2:])
    #         pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
    #         src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

    #         # flatten NxCxHxW to HWxNxC
    #         pos[-1] = pos[-1].permute(2, 0, 1)
    #         src[-1] = src[-1].permute(2, 0, 1)

    #     _, bs, _ = src[0].shape

    #     # QxNxC
    #     query_embed = self.SEG_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
    #     if seg_query is None:
    #         output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
    #     else:
    #         output = seg_query.permute(1, 0, 2)

    #     predictions_SEG_class = []
    #     predictions_class_name_class = []
    #     predictions_region_class = []
    #     predictions_mask = []

    #     # prediction heads on learnable query features
    #     SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(output, mask_features,
    #                                                                                          attn_mask_target_size=
    #                                                                                          size_list[0],
    #                                                                                          SEG_embedding=SEG_embedding,
    #                                                                                          class_name_embedding=class_name_embedding,
    #                                                                                          region_embedding_list = region_embedding_list)
    #     predictions_SEG_class.append(SEG_class)
    #     predictions_class_name_class.append(class_name_class)
    #     predictions_mask.append(outputs_mask)
    #     predictions_region_class.append(region_class_list)

    #     for i in range(self.num_layers):
    #         output = torch.cat([SEG_embedding.transpose(0, 1), output], 0)
    #         SEG_mask = torch.zeros((attn_mask.shape[0], 1, attn_mask.shape[-1]), dtype=torch.bool,
    #                                device=attn_mask.device)
    #         attn_mask = torch.cat([SEG_mask, attn_mask], dim=1)
    #         level_index = i % self.num_feature_levels
    #         attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

    #         # attention: cross-attention first
    #         output = self.transformer_cross_attention_layers[i](
    #             output, src[level_index],
    #             memory_mask=attn_mask,
    #             memory_key_padding_mask=None,  # here we do not apply masking on padded region
    #             pos=pos[level_index], query_pos=query_embed
    #         )

    #         output = self.transformer_self_attention_layers[i](
    #             output, tgt_mask=None,
    #             tgt_key_padding_mask=None,
    #             query_pos=query_embed
    #         )

    #         # FFN
    #         output = self.transformer_ffn_layers[i](
    #             output
    #         )

    #         output = output[1:]
    #         SEG_embedding = output[0].unsqueeze(0).transpose(0, 1)
    #         SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(
    #             output, mask_features,
    #             attn_mask_target_size=
    #             size_list[(
    #                               i + 1) % self.num_feature_levels],
    #             SEG_embedding=SEG_embedding,
    #             class_name_embedding=class_name_embedding,
    #             region_embedding_list=region_embedding_list
    #             )
    #         predictions_SEG_class.append(SEG_class)
    #         predictions_class_name_class.append(class_name_class)
    #         predictions_mask.append(outputs_mask)
    #         predictions_region_class.append(region_class_list)

    #     assert len(predictions_SEG_class) == self.num_layers + 1

    #     out = {
    #         'pred_SEG_logits': predictions_SEG_class[-1],
    #         'pred_class_name_logits': predictions_class_name_class[-1],
    #         'pred_region_logits': predictions_region_class[-1] if predictions_region_class is not None else None,
    #         'pred_masks': predictions_mask[-1],
    #         'aux_outputs': self._set_aux_loss(
    #             predictions_SEG_class, predictions_class_name_class, predictions_mask, predictions_region_class
    #         )
    #     }
    #     return out

    def forward_direct(self, src_flatten, size_list, masks, mask_flatten, spatial_shapes, mask_features, proposal=None, seg_query=None, SEG_embedding=None,
                         class_name_embedding=None, region_embedding_list=None, targets=None, task='pano'):

        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        # src = []
        # pos = []
        # size_list = []

        # disable mask, it does not affect performance
                # disable mask, it does not affect performance
        # masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]

        # src_flatten = []
        # mask_flatten = []
        # spatial_shapes = []
        # for i in range(self.num_feature_levels):
        #     idx=self.num_feature_levels-1-i
        #     bs, c , h, w=x[idx].shape
        #     size_list.append(x[i].shape[-2:])
        #     spatial_shapes.append(x[idx].shape[-2:])
        #     src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
        #     mask_flatten.append(masks[i].flatten(1))


        # src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        # mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        bs = src_flatten.shape[0]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        
        if proposal is None:
            refpoint_embed_undetach = self.SEG_query_embed.weight[None].repeat(bs, 1, 1)
        else:
            refpoint_embed_undetach = proposal
        
        
        if seg_query is None:
            tgt_undetach = self.SEG_query_feat.weight[None].repeat(bs, 1, 1)
        else:
            tgt_undetach = seg_query
        
        # refpoint_embed_stuff = self.query_embed.weight[None].repeat(bs, 1, 1)
        # tgt_stuff = self.query_feat.weight[None].repeat(bs, 1, 1)

        # tgt_undetach=torch.cat([tgt_undetach,tgt_stuff],dim=1)

        SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(tgt_undetach.transpose(0, 1),
                                                                                                        mask_features,
                                                                                                        attn_mask_target_size=
                                                                                                        size_list[0],
                                                                                                        SEG_embedding=SEG_embedding,
                                                                                                        class_name_embedding=class_name_embedding,
                                                                                                        region_embedding_list=region_embedding_list)
        tgt = tgt_undetach.detach()

        # refpoint_embed = [get_bounding_boxes(single_masks) for single_masks in outputs_mask]
        # refpoint_embed = torch.stack(refpoint_embed,dim=0).to(outputs_mask.device)
        # refpoint_embed = box_xyxy_to_cxcywh(refpoint_embed)
        # shape_tensor = torch.as_tensor([outputs_mask.shape[-1], outputs_mask.shape[-2], outputs_mask.shape[-1], outputs_mask.shape[-2]], dtype=torch.float, device=outputs_mask.device)
        # refpoint_embed = refpoint_embed / shape_tensor
        # refpoint_embed = inverse_sigmoid(refpoint_embed)


        # refpoint_embed_undetach=torch.cat([refpoint_embed_undetach,refpoint_embed_stuff],dim=1)
        refpoint_embed = refpoint_embed_undetach.detach()
        # refpoint_embed_undetach = refpoint_embed.detach() # dummy code for interm output

        # if not ('panoptic' in task):
        #     if SEG_class is not None: SEG_class=SEG_class[:,:-self.num_queries_stuff]
        #     if class_name_class is not None: class_name_class=class_name_class[:,:-self.num_queries_stuff]
        #     if region_class_list is not None: region_class_list=[v[:,:-self.num_queries_stuff] for v in region_class_list]
        #     outputs_mask=outputs_mask[:,:-self.num_queries_stuff]
        #     refpoint_embed_undetach=refpoint_embed_undetach[:,:-self.num_queries_stuff]


        interm_outputs={
            'pred_SEG_logits': SEG_class,
            'pred_class_name_logits': class_name_class,
            'pred_region_logits': region_class_list,
            'pred_masks': outputs_mask,
            'pred_boxes': refpoint_embed_undetach.float().sigmoid(),
        }

        tgt_mask = None
        mask_dict = None

        if self.dn != "no" and self.training and targets is not None and class_name_embedding is not None:
            # assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, class_name_embedding, bs, task)
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        # # QxNxC
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # if seg_query is None:
        #     output = self.new_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        # else:
        #     output = seg_query.permute(1, 0, 2)

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_region_class = []
        predictions_mask = []

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(tgt.transpose(0, 1),
                                                                                                mask_features,
                                                                                                attn_mask_target_size=
                                                                                                size_list[0],
                                                                                                SEG_embedding=SEG_embedding,
                                                                                                class_name_embedding=class_name_embedding,
                                                                                                region_embedding_list=region_embedding_list)
            # if not ('panoptic' in task):
            #     if SEG_class is not None: SEG_class=SEG_class[:,:-self.num_queries_stuff]
            #     if class_name_class is not None: class_name_class=class_name_class[:,:-self.num_queries_stuff]
            #     if region_class_list is not None: region_class_list=[v[:,:-self.num_queries_stuff] for v in region_class_list]
            #     outputs_mask=outputs_mask[:,:-self.num_queries_stuff]

            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        # if not ('panoptic' in task):
        #     hs=[hs_[:,:-self.num_queries_stuff] for hs_ in hs]
        #     references=[references_[:,:-self.num_queries_stuff] for references_ in references]
        #     refpoint_embed=refpoint_embed[:,:-self.num_queries_stuff]

        for i, output in enumerate(hs):
            # prediction heads on learnable query features
            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(output.transpose(0, 1),
                                                                                                                    mask_features,
                                                                                                                    attn_mask_target_size=
                                                                                                                    size_list[
                                                                                                                        0],
                                                                                                                    SEG_embedding=SEG_embedding,
                                                                                                                    class_name_embedding=class_name_embedding,
                                                                                                                    region_embedding_list=region_embedding_list)

            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_SEG_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)

        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class_name_class =torch.stack(predictions_class_name_class)

            predictions_class_name_class, out_boxes,predictions_mask =\
                self.dn_post_process(predictions_class_name_class, out_boxes, mask_dict, predictions_mask)
            predictions_class_name_class = list(predictions_class_name_class)
            predictions_mask = list(predictions_mask)


        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_region_logits': predictions_region_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask, predictions_region_class, out_boxes
            ),
            'interm_outputs':interm_outputs
        }
        return out, mask_dict


    def forward_woconcat(self, x, mask_features, proposal=None, seg_query=None, SEG_embedding=None,
                         class_name_embedding=None, region_embedding_list=None, targets=None, task='pano'):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
                # disable mask, it does not affect performance
        masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]

        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))


        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # p_SEG_class, p_class_name_class, p_region_class_list = self.compute_similarity(output_memory, 
        #                                                                                 SEG_embedding=SEG_embedding,
        #                                                                                 class_name_embedding=class_name_embedding,
        #                                                                                 region_embedding_list=region_embedding_list)

        # if p_region_class_list is None:
        #     enc_outputs_class_unselected = p_SEG_class if p_SEG_class is not None else p_class_name_class
        # else:
        #     # tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
        #     # refpoint_embed_undetach = self.query_embed.weight[None].repeat(bs, 1, 1)
        #     enc_outputs_class_unselected = [r_sim.mean(0) for r_sim in p_region_class_list]
        #     enc_outputs_class_unselected = torch.stack(enc_outputs_class_unselected, dim=0)[:,:,None]
        
        # enc_outputs_class_unselected[output_proposals.sum(-1).isinf()] = float("-inf")
        # enc_outputs_coord_unselected = self._bbox_embed(output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
        # topk = self.num_queries
        # topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        # refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
        #                                     topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
        # tgt_undetach = torch.gather(output_memory, 1,
        #                     topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid
        
        if proposal is None:
            refpoint_embed_undetach = self.SEG_query_embed.weight[None].repeat(bs, 1, 1)
        else:
            refpoint_embed_undetach = proposal
        
        
        if seg_query is None:
            tgt_undetach = self.SEG_query_feat.weight[None].repeat(bs, 1, 1)
        else:
            tgt_undetach = seg_query
        
        refpoint_embed_stuff = self.query_embed.weight[None].repeat(bs, 1, 1)
        tgt_stuff = self.query_feat.weight[None].repeat(bs, 1, 1)
        # if seg_query is None:
        #     tgt_stuff = self.query_feat.weight[None].repeat(bs, 1, 1)
        # else:
        # tgt_stuff = seg_query

        tgt_undetach=torch.cat([tgt_undetach,tgt_stuff],dim=1)
        refpoint_embed_undetach=torch.cat([refpoint_embed_undetach,refpoint_embed_stuff],dim=1)
        refpoint_embed = refpoint_embed_undetach.detach()

        SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(tgt_undetach.transpose(0, 1),
                                                                                                        mask_features,
                                                                                                        attn_mask_target_size=
                                                                                                        size_list[0],
                                                                                                        SEG_embedding=SEG_embedding,
                                                                                                        class_name_embedding=class_name_embedding,
                                                                                                        region_embedding_list=region_embedding_list)
        tgt = tgt_undetach.detach()

        # if not ('panoptic' in task):
        #     if SEG_class is not None: SEG_class=SEG_class[:,:-self.num_queries_stuff]
        #     if class_name_class is not None: class_name_class=class_name_class[:,:-self.num_queries_stuff]
        #     if region_class_list is not None: region_class_list=[v[:,:-self.num_queries_stuff] for v in region_class_list]
        #     outputs_mask=outputs_mask[:,:-self.num_queries_stuff]
        #     refpoint_embed_undetach=refpoint_embed_undetach[:,:-self.num_queries_stuff]


        interm_outputs={
            'pred_SEG_logits': SEG_class,
            'pred_class_name_logits': class_name_class,
            'pred_region_logits': region_class_list,
            'pred_masks': outputs_mask,
            'pred_boxes': refpoint_embed_undetach.float().sigmoid()
        }

        tgt_mask = None
        mask_dict = None

        if self.dn != "no" and self.training and targets is not None and class_name_embedding is not None:
            # assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, class_name_embedding, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        # # QxNxC
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # if seg_query is None:
        #     output = self.new_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        # else:
        #     output = seg_query.permute(1, 0, 2)

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_region_class = []
        predictions_mask = []

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(tgt.transpose(0, 1),
                                                                                                mask_features,
                                                                                                attn_mask_target_size=
                                                                                                size_list[0],
                                                                                                SEG_embedding=SEG_embedding,
                                                                                                class_name_embedding=class_name_embedding,
                                                                                                region_embedding_list=region_embedding_list)
            # if not ('panoptic' in task):
            #     if SEG_class is not None: SEG_class=SEG_class[:,:-self.num_queries_stuff]
            #     if class_name_class is not None: class_name_class=class_name_class[:,:-self.num_queries_stuff]
            #     if region_class_list is not None: region_class_list=[v[:,:-self.num_queries_stuff] for v in region_class_list]
            #     outputs_mask=outputs_mask[:,:-self.num_queries_stuff]

            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        # if not ('panoptic' in task):
        #     hs=[hs_[:,:-self.num_queries_stuff] for hs_ in hs]
        #     references=[references_[:,:-self.num_queries_stuff] for references_ in references]
        #     refpoint_embed=refpoint_embed[:,:-self.num_queries_stuff]

        for i, output in enumerate(hs):
            # prediction heads on learnable query features
            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(output.transpose(0, 1),
                                                                                                                    mask_features,
                                                                                                                    attn_mask_target_size=
                                                                                                                    size_list[
                                                                                                                        0],
                                                                                                                    SEG_embedding=SEG_embedding,
                                                                                                                    class_name_embedding=class_name_embedding,
                                                                                                                    region_embedding_list=region_embedding_list)

            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_SEG_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)

        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class_name_class =torch.stack(predictions_class_name_class)

            predictions_class_name_class, out_boxes,predictions_mask =\
                self.dn_post_process(predictions_class_name_class, out_boxes, mask_dict, predictions_mask)
            predictions_class_name_class = list(predictions_class_name_class)
            predictions_mask = list(predictions_mask)


        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_region_logits': predictions_region_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask, predictions_region_class, out_boxes
            ),
            'interm_outputs':interm_outputs
        }
        return out, mask_dict
    
    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list
    
    def compute_similarity(self, output, SEG_embedding=None,
                                class_name_embedding=None, region_embedding_list=None):
        # SEG_embedding = self.SEG_norm(SEG_embedding).expand_as(decoder_output)
        # SEG_embedding = SEG_embedding.expand_as(decoder_output)
        decoder_output = output
        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            if self.seg_norm:
                SEG_embedding = self.SEG_norm(SEG_embedding)
                SEG_embedding = self.seg_proj_after_norm(SEG_embedding)
            SEG_class = torch.einsum('bld,bcd->blc', decoder_seg_output, SEG_embedding)
        else:
            SEG_class = None
        # SEG_class = F.cosine_similarity(decoder_seg_output, SEG_embedding, dim=-1, eps=1e-6).unsqueeze(-1)
        if class_name_embedding is not None:
            # decoder_class_output = decoder_output.detach()
            decoder_class_output = decoder_output
            if self.seg_proj:
                decoder_class_output = self.CLASS_proj(decoder_class_output)
            else:
                decoder_class_output = decoder_class_output
            if self.seg_norm:
                class_name_embedding = self.class_name_norm(class_name_embedding)
                class_name_embedding = self.class_name_proj_after_norm(class_name_embedding)
            dot_product = torch.einsum('bld,bcd->blc', decoder_class_output, class_name_embedding[:,:-1]) # remove bg
            dot_product = torch.nan_to_num(dot_product, nan=-100)
            # dot_product = self.class_embed_sim(decoder_class_output)
            # decoder_output_mag = torch.norm(decoder_output, dim=-1, keepdim=True)
            # class_name_embedding_mag = torch.norm(class_name_embedding, dim=-1, keepdim=True)
            # class_name_class = dot_product / (decoder_output_mag * class_name_embedding_mag.transpose(-1, -2) + 1e-8)
            if self.seg_fuse_score:
                class_SEG_class = SEG_class.expand_as(dot_product)
                reverse_bg_mask = torch.ones_like(class_SEG_class).to(dtype=class_SEG_class.dtype,device=class_SEG_class.device)
                reverse_bg_mask[:,:,-1] = -reverse_bg_mask[:,:,-1]
                class_name_class = dot_product * class_SEG_class * reverse_bg_mask
            else:
                class_name_class = dot_product
        else:
            class_name_class = None

        if region_embedding_list is not None:
            if self.seg_proj:
                decoder_region_output = self.REGION_proj(decoder_output)
            else:
                decoder_region_output = decoder_output
            region_class_list = []
            for sample_decoder_output, region_embedding in zip(decoder_region_output, region_embedding_list):
                sample_region_class = torch.einsum('kd,ld->kl', region_embedding, sample_decoder_output)
                region_class_list.append(sample_region_class)
        else:
            region_class_list = None

        return SEG_class, class_name_class, region_class_list


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, SEG_embedding=None,
                                 class_name_embedding=None, region_embedding_list=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # SEG_embedding = self.SEG_norm(SEG_embedding).expand_as(decoder_output)
        # SEG_embedding = SEG_embedding.expand_as(decoder_output)
        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            if self.seg_norm:
                SEG_embedding = self.SEG_norm(SEG_embedding)
                SEG_embedding = self.seg_proj_after_norm(SEG_embedding)
            SEG_class = torch.einsum('bld,bcd->blc', decoder_seg_output, SEG_embedding)
        else:
            SEG_class = None
        # SEG_class = F.cosine_similarity(decoder_seg_output, SEG_embedding, dim=-1, eps=1e-6).unsqueeze(-1)
        if class_name_embedding is not None:
            # decoder_class_output = decoder_output.detach()
            decoder_class_output = decoder_output
            if self.seg_proj:
                decoder_class_output = self.CLASS_proj(decoder_class_output)
            else:
                decoder_class_output = decoder_class_output
            if self.seg_norm:
                class_name_embedding = self.class_name_norm(class_name_embedding)
                class_name_embedding = self.class_name_proj_after_norm(class_name_embedding)
            dot_product = torch.einsum('bld,bcd->blc', decoder_class_output, class_name_embedding)
            dot_product = torch.nan_to_num(dot_product, nan=-100)
            # dot_product = self.class_embed_sim(decoder_class_output)
            # decoder_output_mag = torch.norm(decoder_output, dim=-1, keepdim=True)
            # class_name_embedding_mag = torch.norm(class_name_embedding, dim=-1, keepdim=True)
            # class_name_class = dot_product / (decoder_output_mag * class_name_embedding_mag.transpose(-1, -2) + 1e-8)
            if self.seg_fuse_score:
                class_SEG_class = SEG_class.expand_as(dot_product)
                reverse_bg_mask = torch.ones_like(class_SEG_class).to(dtype=class_SEG_class.dtype,device=class_SEG_class.device)
                reverse_bg_mask[:,:,-1] = -reverse_bg_mask[:,:,-1]
                class_name_class = dot_product * class_SEG_class * reverse_bg_mask
            else:
                class_name_class = dot_product
        else:
            class_name_class = None
        if region_embedding_list is not None:
            if self.seg_proj:
                decoder_region_output = self.REGION_proj(decoder_output)
            else:
                decoder_region_output = decoder_output
            region_class_list = []
            for sample_decoder_output, region_embedding in zip(decoder_region_output, region_embedding_list):
                sample_region_class = torch.einsum('kd,ld->kl', region_embedding, sample_decoder_output)
                region_class_list.append(sample_region_class)
        else:
            region_class_list = None

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list

    @torch.jit.unused
    def _set_aux_loss(self, outputs_SEG_class, outputs_class_name_class, outputs_seg_masks, predictions_region_class, outputs_boxes):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        #     return [
        #         {"pred_logits": a, "pred_masks": b}
        #         for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        #     ]
        # else:
        #     return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

        return [
            {"pred_SEG_logits": a, "pred_class_name_logits": b, "pred_masks": c, "pred_region_logits": d, "pred_boxes": e}
            for a, b, c, d, e in zip(outputs_SEG_class[:-1], outputs_class_name_class[:-1], outputs_seg_masks[:-1],
                                  predictions_region_class[:-1], outputs_boxes[:-1])
        ]






