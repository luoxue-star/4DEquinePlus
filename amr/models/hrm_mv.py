"""
Multi-view Horse Avatar Reconstruction Model
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
from .hrm import HRM

class MVHRM(HRM):
    def configure_optimizers(self):
        """
        Setup model Optimizers and Learning Rate Scheduler.
        Overrides HRM to use self.parameters() instead of self.get_parameters().
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.cfg.TRAIN.LR}]
        optimizer = torch.optim.AdamW(params=param_groups,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                      fused=False)

        warmup_steps = self.cfg.TRAIN.get('WARMUP_STEPS', 3000)
        total_steps = self.cfg.GENERAL.TOTAL_STEPS
        base_lr_value = float(self.cfg.TRAIN.LR)
        initial_lr = self.cfg.TRAIN.get('INITIAL_LR', 1e-10)
        warmup_start_factor = min(max(initial_lr / max(base_lr_value, 1e-30), 0.0), 1.0)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                s = float(current_step) / float(max(1, warmup_steps))
                return warmup_start_factor + (1.0 - warmup_start_factor) * s
            if total_steps and total_steps > warmup_steps:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        result = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            result["gradient_clip_val"] = self.cfg.TRAIN.GRAD_CLIP_VAL
            result["gradient_clip_algorithm"] = "norm"

        return result

    def forward_backbone(self, image: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward the DINOV3 Backbone
        Args:
            image (torch.Tensor): Image tensor of shape [B, 3, H, W] or [B, S, 3, H, W]
            valid_mask (torch.Tensor): Optional [B, S] bool mask for multi-view padding
        Returns:
            torch.Tensor: Features from the backbone
        """
        img_feats = self.backbone(image, valid_mask=valid_mask)
        return img_feats

    def forward_latent_points(self, query_points: torch.Tensor, image: torch.Tensor, valid_mask: torch.Tensor = None, source_masks: torch.Tensor = None):
        """
        Forward the latent points
        Args:
            query_points (torch.Tensor): Query points tensor of shape [B, N, 3]
            image (torch.Tensor): Image tensor of shape [B, 3, H, W] or [B, S, 3, H, W]
            valid_mask (torch.Tensor): Optional [B, S] bool mask for multi-view padding
            source_masks (torch.Tensor): Optional [B, S, 1, H, W] foreground masks
        Returns:
            torch.Tensor: Latent points tensor of shape [B, N, 1024]
        """
        feats = self.forward_backbone(image, valid_mask=valid_mask)  # [B, S, H/14*W/14+1, 1024]
        cls_feats, image_feats = feats[:, :, -1:, :], feats[:, :, :-1, :]  # [B, S, 1, D], [B, S, N_patches, D]

        # Motion tokens: mean-pool over valid views → [B, D]
        motion_tokens = self.motion_embed_mlp(cls_feats).squeeze(2)  # [B, S, D]
        if valid_mask is not None:
            # Zero out padded views before averaging, divide by real view count
            motion_tokens = motion_tokens * valid_mask.unsqueeze(-1)  # [B, S, D]
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            motion_tokens = motion_tokens.sum(dim=1) / valid_count  # [B, D]
        else:
            motion_tokens = motion_tokens.mean(dim=1)  # [B, D]

        B, S, N_patches, D = image_feats.shape
        need_pos = self.decoder.use_inter_frame_attn
        patch_h = patch_w = int(math.isqrt(N_patches))

        # Build 2D (y, x) positions for RoPE — only when inter-frame attn is on
        if need_pos:
            _ys = torch.arange(patch_h, device=image_feats.device)
            _xs = torch.arange(patch_w, device=image_feats.device)
            grid_pos = torch.stack(
                torch.meshgrid(_ys, _xs, indexing='ij'), dim=-1
            ).reshape(1, 1, N_patches, 2).expand(B, S, N_patches, 2)
            grid_pos_flat = grid_pos.reshape(B, S * N_patches, 2)  # [B, S*N_patches, 2]

        # Filter background tokens using source masks
        filter_bg = self.cfg.MODEL.get("FILTER_BG_TOKENS", True)
        cond_mask = None
        cond_pos = None
        if filter_bg and source_masks is not None:
            # Downsample masks to patch resolution
            masks_reshaped = source_masks.reshape(B * S, 1, source_masks.shape[-2], source_masks.shape[-1])
            # Dilate masks before downsampling to preserve edge information
            dilate_k = self.cfg.MODEL.get("MASK_DILATE_KERNEL", 15)
            if dilate_k > 1:
                padding = dilate_k // 2
                masks_reshaped = F.max_pool2d(masks_reshaped, kernel_size=dilate_k, stride=1, padding=padding)
            masks_flat = F.interpolate(
                masks_reshaped,
                size=(patch_h, patch_w),
                mode='nearest',
            )  # [B*S, 1, patch_h, patch_w]
            masks_flat = masks_flat.reshape(B, S, patch_h * patch_w) > 0.  # [B, S, N_patches] bool

            # # DEBUG: save masks for visualization
            # import os, torchvision
            # debug_dir = "./outputs/debug_05/"
            # os.makedirs(debug_dir, exist_ok=True)
            # step = getattr(self, 'global_step', 0)
            # # Save original source masks [B, S, 1, H, W]
            # for b in range(min(B, 2)):
            #     for s in range(S):
            #         src_mask = source_masks[b, s]  # [1, H, W]
            #         torchvision.utils.save_image(src_mask.float(), f"{debug_dir}/step{step}_b{b}_s{s}_source_mask.png")
            #     # Save downsampled patch-level mask [S, patch_h, patch_w]
            #     patch_mask_vis = masks_flat[b].reshape(S, patch_h, patch_w).float()  # [S, ph, pw]
            #     for s in range(S):
            #         torchvision.utils.save_image(patch_mask_vis[s].unsqueeze(0), f"{debug_dir}/step{step}_b{b}_s{s}_patch_mask.png")
            # if valid_mask is not None:
            #     torch.save(valid_mask.cpu(), f"{debug_dir}/step{step}_valid_mask.pt")
            # # Visualize image_feats via PCA → 3-channel pseudo-color [B, S, N_patches, D]
            # with torch.no_grad():
            #     for b in range(min(B, 2)):
            #         for s in range(S):
            #             feat = image_feats[b, s]  # [N_patches, D]
            #             feat_centered = feat - feat.mean(dim=0, keepdim=True)
            #             # PCA: top-3 components
            #             U, Sv, Vt = torch.linalg.svd(feat_centered, full_matrices=False)
            #             pca3 = U[:, :3] * Sv[:3].unsqueeze(0)  # [N_patches, 3]
            #             # Normalize to [0, 1]
            #             pca3 = pca3 - pca3.min(dim=0, keepdim=True).values
            #             pca3 = pca3 / (pca3.max(dim=0, keepdim=True).values + 1e-8)
            #             pca3 = pca3.reshape(patch_h, patch_w, 3).permute(2, 0, 1)  # [3, ph, pw]
            #             torchvision.utils.save_image(pca3, f"{debug_dir}/step{step}_b{b}_s{s}_feat_pca.png")
            # print(f"[DEBUG] Saved masks & feats to {debug_dir}/ at step {step}, B={B}, S={S}, patch=({patch_h},{patch_w})")

            # Also mask out invalid (padded) views
            if valid_mask is not None:
                masks_flat = masks_flat & valid_mask.unsqueeze(-1)  # [B, S, N_patches]

            # Flatten S*N_patches
            masks_flat = masks_flat.reshape(B, S * N_patches)  # [B, S*N_patches]
            image_feats_flat = image_feats.reshape(B, S * N_patches, D)  # [B, S*N_patches, D]

            def filter_background_tokens(
                image_feats: torch.Tensor,
                token_mask: torch.Tensor,
                token_pos: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                """Select foreground tokens and pad only when batch rows need alignment."""
                batch_size, num_tokens, dim = image_feats.shape
                fg_counts = token_mask.sum(dim=1)
                max_fg = int(fg_counts.max().item())
                if max_fg == 0:
                    filtered_feats = image_feats.new_zeros(batch_size, 1, dim)
                    cond_mask = token_mask.new_zeros(batch_size, 1)
                    filtered_pos = None
                    if token_pos is not None:
                        filtered_pos = token_pos.new_zeros(batch_size, 1, token_pos.shape[-1])
                    return filtered_feats, cond_mask, filtered_pos

                if batch_size == 1:
                    fg_indices = token_mask[0].nonzero(as_tuple=True)[0]
                    filtered_feats = image_feats.index_select(1, fg_indices)
                    filtered_pos = token_pos.index_select(1, fg_indices) if token_pos is not None else None
                    return filtered_feats, None, filtered_pos

                sorted_indices = token_mask.float().argsort(dim=1, descending=True, stable=True)
                gather_indices = sorted_indices[:, :max_fg]
                cond_mask = torch.gather(token_mask, 1, gather_indices)

                filtered_feats = torch.gather(
                    image_feats,
                    1,
                    gather_indices.unsqueeze(-1).expand(batch_size, max_fg, dim),
                )
                filtered_pos = None
                if token_pos is not None:
                    filtered_pos = torch.gather(
                        token_pos,
                        1,
                        gather_indices.unsqueeze(-1).expand(batch_size, max_fg, token_pos.shape[-1]),
                    )

                return filtered_feats, cond_mask, filtered_pos

            filtered_feats, cond_mask, filtered_pos = filter_background_tokens(
                image_feats_flat,
                masks_flat,
                grid_pos_flat if need_pos else None,
            )

            image_feats = filtered_feats  # [B, K, D]
            if need_pos:
                cond_pos = filtered_pos   # [B, K, 2]
        else:
            # No mask: flatten S*N → [B, S*N, D]
            image_feats = image_feats.reshape(B, S * N_patches, D)
            # Mask out patches from padded (invalid) views
            if valid_mask is not None:
                # valid_mask: [B, S] bool → expand to [B, S, N_patches] → [B, S*N_patches]
                cond_mask = valid_mask.unsqueeze(-1).expand(B, S, N_patches).reshape(B, S * N_patches)
            if need_pos:
                cond_pos = grid_pos_flat  # [B, S*N_patches, 2]

        tokens = self.forward_transformer(image_feats,
                                          query_points=query_points,
                                          motion_tokens=motion_tokens,
                                          cond_mask=cond_mask,
                                          cond_pos=cond_pos)
        return tokens, image_feats

    def forward_transformer(self, image_feats: torch.Tensor, query_points: torch.Tensor, motion_tokens: torch.Tensor, cond_mask: torch.Tensor = None, cond_pos: torch.Tensor = None):
        """
        Forward the transformer
        Args:
            image_feats (torch.Tensor): Image features tensor of shape [B, K, 1024] (filtered fg tokens)
            query_points (torch.Tensor): Query points tensor of shape [B, N, 3]
            motion_tokens (torch.Tensor): Motion tokens tensor of shape [B, D]
            cond_mask (torch.Tensor): Optional [B, K] bool mask (True=real, False=pad)
            cond_pos (torch.Tensor): Optional [B, K, 2] 2-D (y, x) positions for RoPE
        Returns:
            torch.Tensor: Transformed query points tensor of shape [B, N, 1024]
        """
        B = image_feats.shape[0]
        x = self.point_embedding(query_points)  # [B, N, D]
        x = self.decoder(x,
                        cond=image_feats,  # [B, K, 1024]
                        temb=motion_tokens,  # [B, D]
                        cond_mask=cond_mask,  # [B, K]
                        cond_pos=cond_pos)  # [B, K, 2]
        return x

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        query_points, varen_data = self.renderer.get_query_points(batch['smal_params'], device=batch['source_rgbs'].device)
        valid_mask = batch.get('source_valid_mask', None)
        source_masks = batch.get('source_masks', None)
        latent_points, image_feats = self.forward_latent_points(query_points, batch['source_rgbs'], valid_mask=valid_mask, source_masks=source_masks)
        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            varen_data=varen_data,
            c2w=batch['c2ws'],
            intrinsic=batch['intrs'],
            height=batch['render_image'].shape[-2],
            width=batch['render_image'].shape[-1],
            background_color=batch['render_bg_colors'],
            train=train,
            additional_features=None,
        )

        N, M = batch['c2ws'].shape[:2]  # c2w: [batch_size, N_view, 4, 4]
        assert (
            render_results["comp_rgb"].shape[0] == N
        ), "Batch size mismatch for render_results"
        assert (
            render_results["comp_rgb"].shape[1] == M
        ), "Number of rendered views should be consistent with render_cameras"

        gs_attrs_list = render_results.pop("gs_attr")

        offset_list = []
        scaling_list = []
        for gs_attrs in gs_attrs_list:
            offset_list.append(gs_attrs.offset_xyz)
            scaling_list.append(gs_attrs.scaling)
        offset_output = torch.stack(offset_list)
        scaling_output = torch.stack(scaling_list)

        return {
            "latent_points": latent_points,
            "offset_output": offset_output,
            "scaling_output": scaling_output,
            **render_results,
        }
