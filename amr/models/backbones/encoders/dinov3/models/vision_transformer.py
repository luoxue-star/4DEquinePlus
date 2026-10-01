# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn
from einops import rearrange

from ..layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from ..utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        alt_start: int = -1,
        cat_token: bool = True,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.alt_start = alt_start
        self.cat_token = cat_token

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def process_attention(self, x, block, attn_type="global", rope=None, attn_mask=None, valid_mask=None):
        """Process attention with local or global scope for multi-view inputs.

        Args:
            valid_mask: [B, S] bool tensor. True = valid view, False = padding view.
                        Only used when attn_type="global" to mask out padding tokens.
        """
        b, s, n = x.shape[:3]
        if attn_type == "local":
            # Local attention: process each view independently
            x = rearrange(x, "b s n c -> (b s) n c")
            if rope is not None:
                # Expand rope for all views
                rope_sin, rope_cos = rope
                if rope_sin.ndim == 3:  # [heads, patches, embed_dim]
                    rope = (rope_sin, rope_cos)
                elif rope_sin.ndim == 4:  # [batch, heads, patches, embed_dim]
                    rope_sin = rope_sin.unsqueeze(1).expand(-1, s, -1, -1, -1)
                    rope_cos = rope_cos.unsqueeze(1).expand(-1, s, -1, -1, -1)
                    rope_sin = rearrange(rope_sin, "b s h n c -> (b s) h n c")
                    rope_cos = rearrange(rope_cos, "b s h n c -> (b s) h n c")
                    rope = (rope_sin, rope_cos)
        elif attn_type == "global":
            # Global attention: process all views together
            x = rearrange(x, "b s n c -> b (s n) c")

            # Build attention mask to exclude padding views
            if valid_mask is not None:
                # valid_mask: [B, S] bool -> expand to per-token mask [B, S*N]
                mask_per_token = valid_mask.unsqueeze(-1).expand(-1, -1, n).reshape(b, s * n)
                # Key mask for sdpa: [B, 1, 1, S*N] (True = attend, False = masked)
                attn_mask = mask_per_token.unsqueeze(1).unsqueeze(2)

            if rope is not None:
                # Concatenate rope for all views
                rope_sin, rope_cos = rope
                if rope_sin.ndim == 3:  # [heads, patches, embed_dim]
                    # Repeat for all views
                    rope_sin = rope_sin.unsqueeze(1).expand(-1, s, -1, -1)
                    rope_cos = rope_cos.unsqueeze(1).expand(-1, s, -1, -1)
                    rope_sin = rearrange(rope_sin, "h s n c -> h (s n) c")
                    rope_cos = rearrange(rope_cos, "h s n c -> h (s n) c")
                    rope = (rope_sin, rope_cos)
                elif rope_sin.ndim == 4:  # [batch, heads, patches, embed_dim]
                    rope_sin = rope_sin.unsqueeze(1).expand(-1, s, -1, -1, -1)
                    rope_cos = rope_cos.unsqueeze(1).expand(-1, s, -1, -1, -1)
                    rope_sin = rearrange(rope_sin, "b s h n c -> b h (s n) c")
                    rope_cos = rearrange(rope_cos, "b s h n c -> b h (s n) c")
                    rope = (rope_sin, rope_cos)
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        # Call block with positional argument, not keyword argument
        x = block(x, rope, attn_mask=attn_mask)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s, n=n)
        return x

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            # Check if input is multi-view
            if t_x.ndim == 5:
                B, S, C, H, W = t_x.shape
                t_x = rearrange(t_x, "b s c h w -> (b s) c h w")
                t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
                t2_x = rearrange(t2_x, "(b s) n c -> b s n c", b=B, s=S)
            else:
                t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)

        local_x = None
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]

            # Process each input in the list
            processed_x = []
            for idx, (t_x, t_rope) in enumerate(zip(x, rope_sincos)):
                is_multiview = t_x.ndim == 4  # [B, S, N, C]

                if is_multiview and self.alt_start != -1:
                    # Alternating local-global attention
                    if i >= self.alt_start and i % 2 == 1:
                        t_x = self.process_attention(t_x, blk, "global", rope=t_rope)
                    else:
                        t_x = self.process_attention(t_x, blk, "local", rope=t_rope)
                        if idx == 0:  # Store local features from first input
                            local_x = t_x
                else:
                    t_x = blk(t_x, t_rope)

                processed_x.append(t_x)

            x = processed_x

        all_x = x
        output = []
        for idx, (x_item, masks) in enumerate(zip(all_x, masks_list)):
            # Check if we have dual token output
            is_dual_token = self.cat_token and local_x is not None and x_item.ndim == 4

            if is_dual_token and idx == 0:
                # Concatenate local and global features for first input
                x_concat = torch.cat([local_x, x_item], dim=-1)
                local_feat = x_concat[..., :self.embed_dim]
                global_feat = x_concat[..., self.embed_dim:]
            else:
                x_concat = x_item
                local_feat = x_item
                global_feat = None

            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(local_feat[..., :self.n_storage_tokens + 1, :])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(local_feat[..., :self.n_storage_tokens + 1, :])
                else:
                    x_norm_cls_reg = self.norm(local_feat[..., :self.n_storage_tokens + 1, :])
                x_norm_patch = self.norm(local_feat[..., self.n_storage_tokens + 1:, :])
            else:
                x_norm = self.norm(local_feat)
                x_norm_cls_reg = x_norm[..., :self.n_storage_tokens + 1, :]
                x_norm_patch = x_norm[..., self.n_storage_tokens + 1:, :]

            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[..., 0, :],
                    "x_storage_tokens": x_norm_cls_reg[..., 1:, :],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x_concat,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1, valid_mask: Tensor = None) -> List[Tensor]:
        # Check if input is multi-view (5D tensor: B, S, C, H, W)
        is_multiview = x.ndim == 5

        if is_multiview:
            B, S, C, H, W = x.shape
            x = rearrange(x, "b s c h w -> (b s) c h w")
            x, hw_tuple = self.prepare_tokens_with_masks(x)
            if self.alt_start != -1:
                # Only reshape to 4D when alternating attention is enabled
                x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
            H_patch, W_patch = hw_tuple
        else:
            x, hw_tuple = self.prepare_tokens_with_masks(x)
            H_patch, W_patch = hw_tuple

        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

        # Store local features for dual token output
        local_x = None

        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H_patch, W=W_patch)
            else:
                rope_sincos = None

            if is_multiview and self.alt_start != -1:
                # Alternating local-global attention for multi-view
                if i >= self.alt_start and i % 2 == 1:
                    # Odd layers after alt_start: global attention
                    x = self.process_attention(x, blk, "global", rope=rope_sincos, valid_mask=valid_mask)
                else:
                    # Even layers or before alt_start: local attention
                    x = self.process_attention(x, blk, "local", rope=rope_sincos)
                    local_x = x
            else:
                # Standard single-view processing
                x = blk(x, rope_sincos)

            if i in blocks_to_take:
                if is_multiview and self.cat_token and local_x is not None:
                    # Concatenate local and global features
                    out_x = torch.cat([local_x, x], dim=-1)
                else:
                    out_x = x
                # Reshape back to multi-view format if processed as flattened batch
                if is_multiview and out_x.ndim == 3:
                    out_x = rearrange(out_x, "(b s) n c -> b s n c", b=B, s=S)
                output.append(out_x)

        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
        valid_mask: torch.Tensor = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n, valid_mask=valid_mask)

        # Check if we have dual token output (concatenated local and global features)
        is_dual_token = outputs[0].shape[-1] == (self.embed_dim * 2)

        if norm:
            outputs_normed = []
            for out in outputs:
                if is_dual_token:
                    # Split local and global features
                    local_feat = out[..., :self.embed_dim]
                    global_feat = out[..., self.embed_dim:]

                    if self.untie_cls_and_patch_norms:
                        # Normalize local features
                        local_cls_reg = local_feat[..., :self.n_storage_tokens + 1, :]
                        local_patch = self.norm(local_feat[..., self.n_storage_tokens + 1:, :])
                        local_normed = torch.cat([local_cls_reg, local_patch], dim=-2)

                        # Normalize global features
                        global_cls_reg = self.cls_norm(global_feat[..., :self.n_storage_tokens + 1, :])
                        global_patch = self.norm(global_feat[..., self.n_storage_tokens + 1:, :])
                        global_normed = torch.cat([global_cls_reg, global_patch], dim=-2)
                    else:
                        # Apply same norm to both
                        local_normed = self.norm(local_feat)
                        global_normed = self.norm(global_feat)

                    # Concatenate normalized features
                    outputs_normed.append(torch.cat([local_normed, global_normed], dim=-1))
                else:
                    # Standard normalization
                    if self.untie_cls_and_patch_norms:
                        x_norm_cls_reg = self.cls_norm(out[..., :self.n_storage_tokens + 1, :])
                        x_norm_patch = self.norm(out[..., self.n_storage_tokens + 1:, :])
                        outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=-2))
                    else:
                        outputs_normed.append(self.norm(out))
            outputs = outputs_normed

        class_tokens = [out[..., 0, :self.embed_dim] for out in outputs]  # Always take from first feature dimension
        extra_tokens = [out[..., 1:self.n_storage_tokens + 1, :self.embed_dim] for out in outputs]
        outputs = [out[..., self.n_storage_tokens + 1:, :] for out in outputs]

        if reshape:
            B, _, h, w = x.shape if x.ndim == 4 else (x.shape[0], x.shape[1], x.shape[3], x.shape[4])
            if is_dual_token:
                # Reshape both local and global features
                outputs = [
                    out.reshape(*out.shape[:-2], h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    if out.ndim == 3 else
                    out.reshape(*out.shape[:-2], h // self.patch_size, w // self.patch_size, -1).permute(0, 1, 4, 2, 3).contiguous()
                    for out in outputs
                ]
            else:
                outputs = [
                    out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    if out.ndim == 3 else
                    out.reshape(B, -1, h // self.patch_size, w // self.patch_size, out.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()
                    for out in outputs
                ]

        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
