# StableSyncNet, from LatentSync (commit a229c39):
# https://github.com/bytedance/LatentSync
#   latentsync/models/stable_syncnet.py and the Attention class of latentsync/models/attention.py
#   (itself adapted from diffusers, Apache-2.0).
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates. Licensed under the Apache License,
# Version 2.0 (see LICENSE in this directory).
#
# Modifications by ForensiX: diffusers' FeedForward (GEGLU) is inlined with the same parameter
# names, so diffusers isn't a dependency; gradient checkpointing is removed.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GEGLU(nn.Module):
    """diffusers.models.activations.GEGLU (Apache-2.0)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    """diffusers.models.attention.FeedForward with activation_fn="geglu" (Apache-2.0)."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, activation_fn: str = "geglu"):
        super().__init__()
        assert activation_fn == "geglu"
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList([GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def split_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    def concat_heads(self, tensor):
        batch_size, heads, seq_len, head_dim = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size, seq_len, heads * head_dim)
        return tensor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        query = self.split_heads(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.split_heads(key)
        value = self.split_heads(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # Use PyTorch native implementation of FlashAttention-2
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        hidden_states = self.concat_heads(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class StableSyncNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = DownEncoder2D(
            in_channels=config["audio_encoder"]["in_channels"],
            block_out_channels=config["audio_encoder"]["block_out_channels"],
            downsample_factors=config["audio_encoder"]["downsample_factors"],
            dropout=config["audio_encoder"]["dropout"],
            attn_blocks=config["audio_encoder"]["attn_blocks"],
                    )

        self.visual_encoder = DownEncoder2D(
            in_channels=config["visual_encoder"]["in_channels"],
            block_out_channels=config["visual_encoder"]["block_out_channels"],
            downsample_factors=config["visual_encoder"]["downsample_factors"],
            dropout=config["visual_encoder"]["dropout"],
            attn_blocks=config["visual_encoder"]["attn_blocks"],
                    )

        self.eval()

    def forward(self, image_sequences, audio_sequences):
        vision_embeds = self.visual_encoder(image_sequences)  # (b, c, 1, 1)
        audio_embeds = self.audio_encoder(audio_sequences)  # (b, c, 1, 1)

        vision_embeds = vision_embeds.reshape(vision_embeds.shape[0], -1)  # (b, c)
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)  # (b, c)

        # Make them unit vectors
        vision_embeds = F.normalize(vision_embeds, p=2, dim=1)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=1)

        return vision_embeds, audio_embeds


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        act_fn: str = "silu",
        downsample_factor=2,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "silu":
            self.act_fn = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

        if isinstance(downsample_factor, list):
            downsample_factor = tuple(downsample_factor)

        if downsample_factor == 1:
            self.downsample_conv = None
        else:
            self.downsample_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=0
            )
            self.pad = (0, 1, 0, 1)
            if isinstance(downsample_factor, tuple):
                if downsample_factor[0] == 1:
                    self.pad = (0, 1, 1, 1)  # The padding order is from back to front
                elif downsample_factor[1] == 1:
                    self.pad = (1, 1, 0, 1)

    def forward(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        hidden_states += input_tensor

        if self.downsample_conv is not None:
            hidden_states = F.pad(hidden_states, self.pad, mode="constant", value=0)
            hidden_states = self.downsample_conv(hidden_states)

        return hidden_states


class AttentionBlock2D(nn.Module):
    def __init__(self, query_dim, norm_num_groups=32, dropout=0.0):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=1e-6, affine=True)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.ff = FeedForward(query_dim, dropout=dropout, activation_fn="geglu")

        self.conv_in = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(query_dim=query_dim, heads=8, dim_head=query_dim // 8, dropout=dropout, bias=True)

    def forward(self, hidden_states):
        assert hidden_states.dim() == 4, f"Expected hidden_states to have ndim=4, but got ndim={hidden_states.dim()}."

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.conv_in(hidden_states)
        hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")

        norm_hidden_states = self.norm2(hidden_states)

        hidden_states = self.attn(norm_hidden_states, attention_mask=None) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=height, w=width).contiguous()
        hidden_states = self.conv_out(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states


class DownEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels=4 * 16,
        block_out_channels=[64, 128, 256, 256],
        downsample_factors=[2, 2, 2, 2],
        layers_per_block=2,
        norm_num_groups=32,
        attn_blocks=[1, 1, 1, 1],
        dropout: float = 0.0,
        act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        # in
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        self.down_blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = block_out_channel

            down_block = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=output_channels,
                downsample_factor=downsample_factors[i],
                norm_num_groups=norm_num_groups,
                dropout=dropout,
                act_fn=act_fn,
            )

            self.down_blocks.append(down_block)

            if attn_blocks[i] == 1:
                attention_block = AttentionBlock2D(query_dim=output_channels, dropout=dropout)
                self.down_blocks.append(attention_block)

        # out
        self.norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.act_fn_out = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)

        # down
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        # post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.act_fn_out(hidden_states)

        return hidden_states
