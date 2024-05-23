# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # scale-level embedding，对4个特征层分别附加d_model维度的embedding，用于区分query对应的具体特征
        # 将torch.Tensor包装成torch.nn.Parameter后，pytorch会自动将其加入到模型的参数泪飚中，在训练时会自动优化
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        经过backbone输出4个不同尺度的特征图srcs，以及这4个特征图对应的masks和位置编码
        srcs: list4  0=[bs,256,H/8,W/8] 1=[bs,256,H/16,W/16] 2=[bs,256,H/32,W/32] 3=[bs,256,H/64,W/64]
        masks: list4  0=[bs,H/8,W/8] 1=[bs,H/16,W/16] 2=[bs,H/32,W/32] 3=[bs,H/64,W/64]
        pos_embeds: list4  0=[bs,256,H/8,W/8] 1=[bs,256,H/16,W/16] 2=[bs,256,H/32,W/32] 3=[bs,256,H/64,W/64]
        query_embed: query embedding 参数 [300, 512]
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        # 为了encoder的输入做准备：将多尺度特征图、各特征图对应的mask、位置编码、各特征图的高宽、各特征图flatten后的起始索引展平
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w) # 特征图的形状
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # (bs, c, h ,w) -> (bs, h*w, c)
            mask = mask.flatten(1) # (bs, h, w) -> (bs, h*w)
            # (bs, c, h ,w) -> (bs, h*w, c)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # 与position embedding相加，且同一特征层，所有query的scale level embedding都是一样的
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        # K =  H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64
        src_flatten = torch.cat(src_flatten, 1)
        # list4[bs, H/8 * W/8] [bs, H/16 * W/16] [bs, H/32 * W/32] [bs, H/64 * W/64] -> [bs, K]
        mask_flatten = torch.cat(mask_flatten, 1)
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # [4, h+w]  4个特征图的高和宽
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 不同层开始的编号
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 各尺度特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # 默认执行
            # 随机初始化 query_embed = nn.Embedding(num_queries, hidden_dim*2)
            # [300, 512] -> [300, 256] + [300, 256]
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            # 初始化query pos [300, 256] -> [bs, 300, 256]
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # 初始化query embedding [300, 256] -> [bs, 300, 256]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # 由query pos接一个全连接层，再归一化后的参考点中心坐标 [bs, 300, 256] -> [bs, 300, 2]
            reference_points = self.reference_points(query_embed).sigmoid()
            # 初始化的归一化参考坐标
            init_reference_out = reference_points

        # decoder
        # tgt: 初始化query embedding [bs, 300, 256]
        # reference_points: 由query pos接一个全连接层 再归一化后的参考点中心坐标 [bs, 300, 2]
        # query_embed: query pos[bs, 300, 256]
        # memory: Encoder输出结果 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        # spatial_shapes: [4, 2] 4个特征层的shape
        # level_start_index: [4, ] 4个特征层flatten后的开始index
        # valid_ratios: [bs, 4, 2]
        # mask_flatten: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 300, 256]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 300, 2]
        #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        reference_points: 4个flatten后特征图对应的归一化参考点坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        # self attention + add + norm
        # query = flatten后的多尺度特征图 + scale-level pos
        # key = 采样点  每个特征点对应周围的4个可学习的采样点
        # value = flatten后的多尺度特征图

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        生成参考点 reference points 威神恶魔参考点是中心点，为什么要归一化
        spatial_shapes: 4个特征图的shape [4,2]
        valid_ratios: 4个特征图中非padding部分的变长占其边的比例 [bs, 4, 2]
        device: cuda
        """
        reference_points_list = []
        # 遍历4个特征图的shape 比如 H_ = 100, W_ = 150
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 0.5 -> 99.5 取100个点 0.5 1.5 2.5 ... 99.5
            # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
            # ref_y: [100, 150] 第一行：150个0.5 第二行：150个1.5 …… 第100行：150个99.5
            # ref_x: [100, 150] 第一行 0.5 1.5 …… 149.5 100行全部相同

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # [100, 150] -> [bs, 15000] 150个0.5 + 150个1.5 + …… + 150个99.5 -> 除以100归一化 
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [100, 150] -> [bs, 15000] 100个 0.5 1.5 …… 99.5 -> 除以150归一化
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # [bs, 15000, 2] 每一项都是xy
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
        # [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
        # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
        # 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 4个flatten后特征图的归一化参考点坐标
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        output = src
        # 4个flatten后特征图的归一化参考点坐标 每个特征点有4个参考点 xy坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        # 经过6层encoder增强后的新特征  每一层不断学习特征层中每个位置和4个采样点的相关性，最终输出的特征是增强后的特征图
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        """
        tgt: 预设的query embedding [bs, 300, 256]
        qeury_pos: 预设的query pos [bs, 300, 256]
        refernce_points: query pos通过一个全连接层->2维 [bs, 300, 4, 2] = [bs, num_query, n_layer, 2]
                         iterative bounding box refinement时 [bs, num_query, n_layer, 4]
        src: 第一层是encoder的memory输出 第2-6层都是上一层输出的output
        src_spatial_shapes: [4, 2] 4个特征图的shape
        src_level_start_index: [4,] 4个特征图flatten后的起始索引
        src_padding_mask: 4个特征层flatten后的mask [bs, H/8*W/8 + H/16*W/16 + H/32*W/32 + H/64*W/64]    
        """
        # self attention
        # query embedding + query pos
        q = k = self.with_pos_embed(tgt, query_pos)
        # 第一个attention的目标：学习各个物体之间的关系/位置 可以知道图像当中哪些位置会存在物体 物体信息->tgt
        # 所有qk都是query embedding + query pos， v就是query embedding
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        # 使用多尺度的可变形注意力模块代替原先的transformer的交叉注意力
        # 第二个attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中，更好地表征了图像中的的物体
        # q=query embedding + query pos, k = query pos 通过一个全连接层->2维, v=encoder的memory输出
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        # [bs, 300, 256]  self-attention输出特征 + cross-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的位置关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """
        tgt: 预设的query embedding [bs, 300, 256]
        query_pos: 预设的query pos [bs, 300, 256]
        reference_points: query pos 通过一个全连接层->2维 [bs, 300, 2]
        src: encoder最后输出的特征即memory [bs, H/8*W/8 + H/16*W/16 + H/32*W/32 + H/64*W/64, 256]
        src_spatial_shapes: 4个特征图的shape [4, 2]
        src_level_start_index: 4个特征图flatten后的起始索引 [4,]
        src_padding_mask: 4个特征层flatten后的mask [bs, H/8*W/8 + H/16*W/16 + H/32*W/32 + H/64*W/64]
        """
        output = tgt

        intermediate = [] # 中间各层+首尾两层=6层输出的解码结果
        intermediate_reference_points = [] # 中间各层+首尾两层输出的参考点（不断矫正）
        for lid, layer in enumerate(self.layers):
            # 得到参考点坐标
            # two stage
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                # one stage模式下参考点是query pos通过一个全连接层线性变换为2为的中心坐标形式(x, y)
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            # decoder layer 
            # output: [bs, 300, 256] = self-attention和cross-attention的输出特征
            # 知道图像中物体与物体之间的关系 + encoder增强后的图像特征 + 图像与物体之间的关系
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            # 使用iterative bounding box refinement 这里的self.bbox_embe就不是None
            # 如果没有iterative bounding box refinement那么reference points是不变的
            # 每层参考点都会根据上一层的输出结果进行矫正
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output) # [bs, 300, 256] -> [bs. 300, 4(xywh)]
                if reference_points.shape[-1] == 4: # two stage
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else: # one stage
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    # 根据decoder每层解码的特征图->回归图（不共享参数） 得到相对参考点的偏移量xy
                    # 然后再加上参考点坐标（反归一化），再进行sigmoid归一化 得到矫正的参考点
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # detach取消了梯度 因为这个参考点在各层中相当于作为先验的角色
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # 默认返回6个decoder层输出一起计算loss
        if self.return_intermediate:
            # 0 [6, bs, 300, 256] 6层decoder输出
            # 1 [6, bs, 300, 2] 6层decoder学习到的参考点归一化中心坐标 一般6层是相同的
            # 但是如果是iterative bounding box refinement那么每层的参考点都会根据上一层的输出结果进行矫正
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


