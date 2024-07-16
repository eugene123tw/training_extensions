"""by lyuwenyu
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from otx.algo.detection.heads import RTDETRTransformer


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

class DetrMaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, inter_dims[1]), inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(min(8, inter_dims[2]), inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(min(8, inter_dims[3]), inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, bbox_mask: torch.Tensor, fpns: list[torch.Tensor]):
        # here we concatenate x, the projected feature map, of shape (batch_size, d_model, heigth/32, width/32) with
        # the bbox_mask = the attention maps of shape (batch_size, n_queries, n_heads, height/32, width/32).
        # We expand the projected feature map to match the number of heads.
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = nn.functional.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = nn.functional.relu(x)

        x = self.out_lay(x)
        return x


class DetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(
        self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: torch.Tensor | None = None):
        """_summary_

        Args:
            q (_type_): object queries tensor of shape (batch_size, num_queries, hidden_dim)
            k (_type_): key (feature map) tensor of shape (batch_size, hidden_dim, height/32, width/32)
            mask (torch.Tensor | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        q = self.q_linear(q)
        k = nn.functional.conv2d(
            k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias
        )
        queries_per_head = q.view(
            q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads
        )
        keys_per_head = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum(
            "bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head
        )

        if mask is not None:
            weights.masked_fill_(
                mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min
            )
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


class RTDETRInstSegTransformer(RTDETRTransformer):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        position_embed_type="sine",
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            position_embed_type=position_embed_type,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_decoder_points=num_decoder_points,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
        )

        self.bbox_attention = DetrMHAttentionMap(
            self.hidden_dim, self.hidden_dim, self.nhead
        )

        self.mask_head = DetrMaskHeadSmallConv(
            self.hidden_dim + self.nhead,
            feat_channels[::-1][-3:],
            self.hidden_dim,
        )

    def forward(self, encoded_feats, feats, last_proj_feat, targets=None):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(encoded_feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, output_memory = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_class,
            denoising_bbox_unact,
        )

        # decoder
        out_bboxes, out_logits, sequence_output = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_sequence, sequence_output = torch.split(sequence_output, dn_meta["dn_num_split"], dim=1)


        # mask head
        batch_size, num_pred, _ = sequence_output.size()

        # get last memory
        f_h, f_w = spatial_shapes[-1]
        last_memory = output_memory[:, level_start_index[-1]:, :].permute(0, 2, 1).view(-1, self.hidden_dim, f_h, f_w)

        # bbox_mask is of shape (batch_size, num_queries, number_of_attention_heads in bbox_attention, height/32, width/32)
        bbox_mask = self.bbox_attention(sequence_output, last_memory)

        # mask_head is of shape (batch_size, 1, height/32, width/32)
        seg_masks = self.mask_head(
            last_proj_feat,
            bbox_mask,
            [feats[2], feats[1], feats[0]],
        )

        pred_masks = seg_masks.view(
            batch_size, num_pred, seg_masks.shape[-2], seg_masks.shape[-1]
        )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1], "pred_masks": pred_masks}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["aux_outputs"].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]
