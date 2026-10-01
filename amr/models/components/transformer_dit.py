from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

assert hasattr(F, "scaled_dot_product_attention"), print(
    "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
)

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


class QwenPointImageAttnProcessor2_0:
    """Attention processor aligning points (sample) with image (context) streams using Qwen logic."""

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenPointImageAttnProcessor2_0 requires PyTorch 2.0 or newer."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Points stream
        encoder_hidden_states: torch.FloatTensor = None,  # Image stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenPointImageAttnProcessor2_0 requires encoder_hidden_states (image stream)."
            )

        seq_image = encoder_hidden_states.shape[1]

        # Points projections (sample branch)
        pts_query = attn.to_q(hidden_states)
        pts_key = attn.to_k(hidden_states)
        pts_value = attn.to_v(hidden_states)

        # Image projections (context branch)
        img_query = attn.add_q_proj(encoder_hidden_states)
        img_key = attn.add_k_proj(encoder_hidden_states)
        img_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape to multi-head format
        pts_query = pts_query.unflatten(-1, (attn.heads, -1))
        pts_key = pts_key.unflatten(-1, (attn.heads, -1))
        pts_value = pts_value.unflatten(-1, (attn.heads, -1))

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        # QK norm if available
        if attn.norm_q is not None:
            pts_query = attn.norm_q(pts_query)
        if attn.norm_k is not None:
            pts_key = attn.norm_k(pts_key)
        if getattr(attn, "norm_added_q", None) is not None:
            img_query = attn.norm_added_q(img_query)
        if getattr(attn, "norm_added_k", None) is not None:
            img_key = attn.norm_added_k(img_key)

        # Optional rotary embeddings (unused but kept for API parity)
        if image_rotary_emb is not None:
            pts_freqs, img_freqs = image_rotary_emb
            pts_query = apply_rotary_emb_qwen(pts_query, pts_freqs, use_real=False)
            pts_key = apply_rotary_emb_qwen(pts_key, pts_freqs, use_real=False)
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)

        # Concatenate streams: [image(context), points(sample)]
        joint_query = torch.cat([img_query, pts_query], dim=1)
        joint_key = torch.cat([img_key, pts_key], dim=1)
        joint_value = torch.cat([img_value, pts_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        img_attn_output = joint_hidden_states[:, :seq_image, :]
        pts_attn_output = joint_hidden_states[:, seq_image:, :]

        pts_attn_output = attn.to_out[0](pts_attn_output)
        if len(attn.to_out) > 1:
            pts_attn_output = attn.to_out[1](pts_attn_output)

        img_attn_output = attn.to_add_out(img_attn_output)

        return pts_attn_output, img_attn_output


class QwenMMJointTransformerBlock(nn.Module):
    """Qwen-inspired dual-stream DiT block for joint point-image conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = "rms_norm",
    ):
        super().__init__()

        if context_pre_only:
            raise ValueError("QwenMMJointTransformerBlock does not support context_pre_only=True")

        attention_head_dim = dim // num_heads
        assert attention_head_dim * num_heads == dim

        self.dim = dim
        self.num_heads = num_heads

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.pts_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.pts_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenPointImageAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )

        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.pts_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.pts_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _apply_modulation(self, x: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = params.chunk(3, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        cond_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): Input hidden states. Query Points features
            encoder_hidden_states (torch.Tensor): Encoder hidden states. Context features
            temb (torch.Tensor): Motion embedding
            cond_mask (torch.Tensor, optional): Bool mask [B, L_cond] (True=real, False=pad)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the updated hidden states and encoder hidden states.
        """

        if temb is None:
            raise ValueError("temb must be provided for QwenMMJointTransformerBlock")

        if temb.shape[-1] != self.dim:
            raise ValueError(
                f"temb dimension {temb.shape[-1]} does not match transformer dimension {self.dim}"
            )

        pts_mod_params = self.pts_mod(temb)
        img_mod_params = self.img_mod(temb)

        pts_mod1, pts_mod2 = pts_mod_params.chunk(2, dim=-1)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)

        pts_normed = self.pts_norm1(hidden_states)
        pts_modulated, pts_gate1 = self._apply_modulation(pts_normed, pts_mod1)

        img_normed = self.img_norm1(encoder_hidden_states)
        img_modulated, img_gate1 = self._apply_modulation(img_normed, img_mod1)

        # Build joint attention mask from cond_mask
        attention_mask = None
        if cond_mask is not None:
            B = cond_mask.shape[0]
            seq_pts = hidden_states.shape[1]
            # Joint key order: [img_tokens, pts_tokens]
            # cond_mask: [B, K] bool, pts always attend
            pts_mask = cond_mask.new_ones(B, seq_pts)  # [B, N_pts]
            joint_mask = torch.cat([cond_mask, pts_mask], dim=1)  # [B, K+N_pts]
            attention_mask = joint_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K+N_pts]

        # TODO: add image_rotary_emb
        attn_hidden, attn_context = self.attn(
            hidden_states=pts_modulated,
            encoder_hidden_states=img_modulated,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + pts_gate1.unsqueeze(1) * attn_hidden
        encoder_hidden_states = encoder_hidden_states + img_gate1.unsqueeze(1) * attn_context

        pts_normed2 = self.pts_norm2(hidden_states)
        pts_modulated2, pts_gate2 = self._apply_modulation(pts_normed2, pts_mod2)
        pts_ff = self.pts_mlp(pts_modulated2)
        hidden_states = hidden_states + pts_gate2.unsqueeze(1) * pts_ff

        img_normed2 = self.img_norm2(encoder_hidden_states)
        img_modulated2, img_gate2 = self._apply_modulation(img_normed2, img_mod2)
        img_ff = self.img_mlp(img_modulated2)
        encoder_hidden_states = encoder_hidden_states + img_gate2.unsqueeze(1) * img_ff

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp_(min=-65504, max=65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clamp_(min=-65504, max=65504)

        return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# Inter-frame self-attention components (LHM++-style VGGT global attention)
# ---------------------------------------------------------------------------

from typing import Dict

class _LayerScale(nn.Module):
    """Per-channel learnable scaling (from CaiT / DeiT-III)."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding (same algorithm as VGGT)."""

    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = frequency
        self.frequency_cache: Dict[tuple, tuple] = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return tokens * cos + self._rotate_features(tokens) * sin

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, num_heads, N, head_dim]  (head_dim must be divisible by 4)
            positions: [B, N, 2]  (y, x coordinates)
        """
        feature_dim = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_pos, tokens.device, tokens.dtype
        )
        vert, horiz = tokens.chunk(2, dim=-1)
        vert = self._apply_1d_rope(vert, positions[..., 0], cos_comp, sin_comp)
        horiz = self._apply_1d_rope(horiz, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vert, horiz), dim=-1)


class InterFrameSelfAttention(nn.Module):
    """
    self-attention block for inter-frame interaction.
    - Pre-norm + custom Attention (QK norm, 2D RoPE, SDPA)
    - LayerScale (init_values=0.01)
    - Mlp (GELU, ratio=4) with LayerScale
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
        qk_norm: bool = True,
        rope_freq: float = 100.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # --- Attention ---
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.ls1 = _LayerScale(dim, init_values=init_values)

        # 2D RoPE
        self.rope = _RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None

        # --- FFN ---
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.ls2 = _LayerScale(dim, init_values=init_values)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, K, D]   image tokens (all views flattened).
            mask: [B, K]      bool (True=real token, False=padding).
            pos:  [B, K, 2]   2-D (y, x) patch positions for RoPE.
        Returns:
            [B, K, D] updated image tokens.
        """
        # --- Self-attention ---
        residual = x
        x = self.norm1(x)

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, H, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Padding mask → SDPA bool mask [B, 1, 1, K]  (True = attend)
        attn_mask = mask[:, None, None, :] if mask is not None else None

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = residual + self.ls1(x)

        # --- FFN ---
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.ls2(x)

        return x
