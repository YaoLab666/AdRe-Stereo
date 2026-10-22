"""Three-branch Test-Time Geometry Adapter (TTGA) from AdRe-Stereo."""

import torch
import torch.nn as nn


class LowRankResidual(nn.Module):
    """Low-rank 3-D residual correction, ``delta C_lr`` in the paper."""
    
    def __init__(self, channels=8, rank=8, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.rank = rank
        
        # Bottleneck design for efficiency
        self.down = nn.Conv3d(channels, rank, kernel_size=1, bias=False)
        
        self.mid = nn.Sequential(
            nn.Conv3d(rank, rank, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        
        self.up = nn.Conv3d(rank, channels, kernel_size=1, bias=False)
        
        # Initialize conservatively to ensure identity mapping at start
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce near-zero output initially."""
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.mid[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W] geometric encoding volume
        Returns:
            residual adaptation: [B, C, D, H, W]
        """
        return self.up(self.mid(self.down(x)))


class SpatialResidual(nn.Module):
    """Image-conditioned spatial residual correction, ``delta C_sp``."""
    
    def __init__(self, feat_channels=96, channels=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, channels, 1, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        """Start as a zero residual while keeping the branch trainable."""
        nn.init.kaiming_normal_(
            self.proj[0].weight, mode='fan_out', nonlinearity='relu'
        )
        # The old low-rank-only checkpoint has no spatial parameters.  A zero
        # output projection therefore preserves its prediction on the first
        # warm-start forward pass.  The projection itself receives gradients
        # immediately and subsequently unlocks gradients to the first layer.
        nn.init.zeros_(self.proj[2].weight)

    def forward(self, feat2d):
        """
        Args:
            feat2d: [B, C2, H, W] spatial features
        Returns:
            spatial bias: [B, C, 1, H, W]
        """
        # Project 2D features and expand to disparity dimension
        proj = self.proj(feat2d)  # [B, C, H, W]
        return proj.unsqueeze(2)  # [B, C, 1, H, W]


class DisparityAxisGate(nn.Module):
    """Depthwise-separable disparity-axis reweighting gate, ``G_d``."""
    
    def __init__(self, channels=8, initial_gate=0.99):
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        self.initial_gate = float(initial_gate)
        self.net = nn.Sequential(
            # Depth-wise conv along disparity axis
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), 
                     padding=(1, 0, 0), groups=channels, bias=False),
            # Pointwise conv for combining features
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),  # Gate between [0, 1]
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize to a spatially constant, near-identity gate."""
        depthwise = self.net[0]
        pointwise = self.net[1]

        # Preserve each channel along the disparity axis.  The pointwise
        # layer initially ignores this signal, but receives gradients on the
        # first update; this avoids the dead branch produced by zeroing both
        # convolutions.
        nn.init.zeros_(depthwise.weight)
        with torch.no_grad():
            depthwise.weight[:, 0, 1, 0, 0] = 1.0

        nn.init.zeros_(pointwise.weight)
        initial_logit = torch.logit(
            torch.tensor(self.initial_gate, dtype=pointwise.bias.dtype)
        )
        nn.init.constant_(pointwise.bias, initial_logit.item())

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            gate: [B, C, D, H, W] in (0, 1]
        """
        return self.net(x)


class TestTimeGeometryAdapter(nn.Module):
    """AdRe-Stereo's low-rank, spatial, and disparity-gating TTGA module."""
    
    def __init__(self,
                 geo_channels=8,
                 feat_channels=96,
                 adapter_rank=8,
                 use_disp_branch=True,
                 use_spatial_branch=True,
                 adapter_dropout=0.0):
        super().__init__()
        
        self.geo_channels = geo_channels
        self.feat_channels = feat_channels
        self.use_disp_branch = use_disp_branch
        self.use_spatial_branch = use_spatial_branch
        
        # Core low-rank adaptation
        self.low_rank_residual = LowRankResidual(
            geo_channels, adapter_rank, adapter_dropout
        )
        
        # Optional spatial branch
        if use_spatial_branch:
            self.spatial_residual = SpatialResidual(feat_channels, geo_channels)
        else:
            self.spatial_residual = None
        
        # Optional disparity-axis reweighting
        if use_disp_branch:
            self.disparity_gate = DisparityAxisGate(geo_channels)
        else:
            self.disparity_gate = None
        
        # Learnable weights for branch mixing
        self.a_lr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.a_sp = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        
        # Initialization for stability
        self._init_params()

    def _init_params(self):
        """Initialize mixture parameters conservatively."""
        nn.init.constant_(self.a_lr, 0.1)
        nn.init.constant_(self.a_sp, 0.1)

    def forward(self, geo_volume, feat2d, high_error_mask=None, return_aux=False):
        """
        Adapt geometric encoding volume with spatial and disparity-based modifications.
        
        Args:
            geo_volume: [B, C, D, H, W] original geometric encoding volume
            feat2d: [B, C2, H, W] spatial features from feature extractor
            high_error_mask: [B, 1, H, W] mask of selected high-error tiles
            return_aux: whether to return auxiliary information
            
        Returns:
            adapted_volume: [B, C, D, H, W] adapted geometric volume
            aux_dict: dict with auxiliary statistics (if return_aux=True)
        """
        
        # Low-rank residual adaptation
        delta_low_rank = self.low_rank_residual(geo_volume)
        
        # Spatial feature augmentation
        spatial_term = 0.0
        if self.spatial_residual is not None:
            spatial_aug = self.spatial_residual(feat2d)  # [B, C, 1, H, W]
            # Expand to full disparity dimension
            spatial_term = spatial_aug.expand_as(geo_volume)
        
        # Disparity-axis reweighting gate
        disp_gate = 1.0
        if self.disparity_gate is not None:
            disp_gate = self.disparity_gate(geo_volume)
        
        # Combine all adaptations
        out = geo_volume + self.a_lr * delta_low_rank + self.a_sp * spatial_term
        out = out * disp_gate
        
        # Sparse inner-loop gate: adapt only selected high-error tiles.
        if high_error_mask is not None:
            # Expand 2D mask to 3D
            high_error_mask_3d = high_error_mask.unsqueeze(2).expand_as(out)
            # Preserve the raw GEV outside selected tiles.
            out = geo_volume * (1.0 - high_error_mask_3d) + out * high_error_mask_3d
        
        if not return_aux:
            return out, {}
        
        # Compute auxiliary statistics for analysis
        aux = {
            "delta_mean": delta_low_rank.abs().mean().detach(),
            "delta_std": delta_low_rank.std().detach(),
            "disp_gate_mean": disp_gate.mean().detach()
            if torch.is_tensor(disp_gate)
            else geo_volume.new_tensor(1.0),
            "a_lr": self.a_lr.detach(),
            "a_sp": self.a_sp.detach(),
        }
        
        return out, aux
