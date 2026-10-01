import torch
import torch.nn as nn
from accelerate.logging import get_logger
from yacs.config import CfgNode
from torchvision import transforms


logger = get_logger(__name__)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        inner_channels, 
        use_clstoken=False,
        out_channel=1024,
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in inner_channels
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.output_conv = nn.Conv2d(sum(inner_channels) , out_channel, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            # Handle multi-view input: [B, S, N, C] -> [B*S, N, C]
            if x.ndim == 4:
                B, S, N, C = x.shape
                x = x.reshape(B * S, N, C)
                batch_size = B * S
            else:
                batch_size = x.shape[0]

            x = x.permute(0, 2, 1).reshape((batch_size, x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)

            out.append(x)

        fusion_feats = torch.cat(out, dim=1)

        fusion_feats = self.output_conv(fusion_feats)

        return fusion_feats


class Dinov3FusionWrapper(nn.Module):
    """
    Dinov3FusionWrapper using original implementation, hacked with modulation.
    """
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.model_name = cfg.MODEL.BACKBONE.get('MODEL_NAME', 'dinov3_vitl16')
        self.modulation_dim = cfg.MODEL.BACKBONE.get('MODULATION_DIM', None)
        model_weight = cfg.MODEL.BACKBONE.get('MODEL_WEIGHT', None)
        self.use_multiview = cfg.MODEL.BACKBONE.get('USE_MULTIVIEW', False)
        self.alt_start = cfg.MODEL.BACKBONE.get('ALT_START', -1)
        self.cat_token = cfg.MODEL.BACKBONE.get('CAT_TOKEN', False)
        self.return_cls_token = cfg.MODEL.BACKBONE.get('RETURN_CLS_TOKEN', False)

        self.model = self._build_dinov3(
            self.model_name,
            modulation_dim=self.modulation_dim,
            pretrained=True,
            model_weight=model_weight,
            alt_start=self.alt_start,
            cat_token=self.cat_token
        )

        # Original layer indices for single-view
        self.intermediate_layer_idx_info = {
            'dinov3_vits16': [2, 5, 8, 11],
            'dinov3_vitb16': [2, 5, 8, 11],
            'dinov3_vitl16': [4, 11, 17, 23],
            'dinov3_vitg16': [9, 19, 29, 39]
        }

        # DA3-style layer indices for multi-view (more dense sampling in later layers)
        self.intermediate_layer_idx_info_mv = {
            'dinov3_vits16': [5, 7, 9, 11],
            'dinov3_vitb16': [5, 7, 9, 11],
            'dinov3_vitl16': [11, 15, 19, 23],
            'dinov3_vitg16': [19, 27, 33, 39]
        }

        # Choose layer indices based on multi-view mode
        if self.use_multiview:
            self.intermediate_layer_idx = self.intermediate_layer_idx_info_mv[self.model_name]
        else:
            self.intermediate_layer_idx = self.intermediate_layer_idx_info[self.model_name]

        # When cat_token=True AND alternating attention is active, feature dimension is doubled (local + global)
        use_cat_token = self.cat_token and self.alt_start != -1
        dpt_in_channels = self.model.embed_dim * 2 if use_cat_token else self.model.embed_dim
        self.fusion_head = DPTHead(in_channels=dpt_in_channels,
                                   inner_channels=[dpt_in_channels] * 4,
                                   out_channel=cfg.MODEL.BACKBONE.get('ENCODER_FEAT_DIM', 1024))

        if cfg.MODEL.BACKBONE.get('FREEZE', True):
            if self.modulation_dim is not None:
                raise ValueError("Modulated Dinov3 requires training, freezing is not allowed.")
            self._freeze()

        self.transforms = self.make_transform()

    def make_transform(self):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([normalize])

    def _freeze(self):
        # logger.warning(f"======== Freezing Dinov3FusionWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov3(model_name: str, modulation_dim: int = None, pretrained: bool = True, model_weight: str = None, alt_start: int = -1, cat_token: bool = True):
        assert model_weight is not None, "Model weight is required for DINOV3"
        from importlib import import_module
        dinov3_hub = import_module(".dinov3.hub.backbones", package=__package__)
        model_fn = getattr(dinov3_hub, model_name)
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained,
                         weights=model_weight, alt_start=alt_start, cat_token=cat_token)
        return model

    # @torch.compile  # disabled: multi-view attention creates >2^31-element tensors, hitting Triton int64/uint32 bug
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None, valid_mask: torch.Tensor = None):
        # image: [N, C, H, W] or [N, S, C, H, W] for multi-view
        # mod: [N, D] or None
        # valid_mask: [N, S] bool or None (True = valid view, False = padding)
        # RGB image with [0,1] scale and properly sized

        is_multiview = image.ndim == 5
        if is_multiview:
            N, S, C, H, W = image.shape
            image = image.reshape(N * S, C, H, W)

        image = self.transforms(image)
        patch_h, patch_w = image.shape[-2] // self.model.patch_size, image.shape[-1] // self.model.patch_size

        if is_multiview:
            image = image.reshape(N, S, C, image.shape[-2], image.shape[-1])

        features = self.model.get_intermediate_layers(image, n=self.intermediate_layer_idx, return_class_token=True, valid_mask=valid_mask)

        # Extract cls tokens if needed
        cls_tokens = []
        patch_features = []
        for feat in features:
            if isinstance(feat, tuple):
                patch_feat, cls_token = feat
                patch_features.append((patch_feat,))
                cls_tokens.append(cls_token)
            else:
                patch_features.append(feat)

        out_local = self.fusion_head(patch_features, patch_h, patch_w)

        out_global = None
        if self.return_cls_token and cls_tokens:
            # Use the last layer's cls token
            out_global = cls_tokens[-1]
            if is_multiview and out_global.ndim == 3:
                # [N, S, D] -> [N*S, D]
                out_global = out_global.reshape(N * S, -1)

        if out_global is not None:
            ret = torch.cat([out_local.permute(0, 2, 3, 1).flatten(1, 2), out_global.unsqueeze(1)], dim=1)
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)

        if is_multiview:
            # Reshape back to [N, S, ...]
            ret = ret.reshape(N, S, *ret.shape[1:])

        return ret