import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, Feature
from core.geometry import CombinedGeometryEncodingVolume
from core.submodule import *
from core.ttga import TestTimeGeometryAdapter
from core.cache import StereoPairCache


autocast = torch.cuda.amp.autocast

class GeometryVolumeRegularizer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class AdReStereo(nn.Module):
    """IGEV backbone augmented with the three-branch TTGA."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = GeometryVolumeRegularizer(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        self.ttga = TestTimeGeometryAdapter(
            geo_channels=8,
            feat_channels=96,
            adapter_rank=getattr(args, "ttga_rank", 8),
            use_disp_branch=getattr(args, "ttga_disparity_gate", True),
            use_spatial_branch=getattr(args, "ttga_spatial_branch", True),
            adapter_dropout=getattr(args, "ttga_dropout", 0.0),
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp

    def encode_pair(self, image1, image2):
        """
        Extract features and build static cost volume without GRU refinement.
        
        Returns:
            cache: StereoPairCache with all intermediate results
        """
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Context features for GRU
            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        cache = StereoPairCache({
            "features_left": features_left,
            "stem_2x": stem_2x,
            "match_left": match_left,
            "match_right": match_right,
            "geo_volume_raw": geo_encoding_volume,
            "net_list": net_list,
            "inp_list": inp_list,
        })
        
        return cache

    def apply_ttga(self, geo_volume_raw, features_left, high_error_mask=None, return_aux=False):
        """
        Apply TTGA to the geometry encoding volume.
        Handles mixed precision by converting inputs to float32 for adapter.
        """
        if self.ttga is None:
            return geo_volume_raw, {}
        
        # Convert to float32 for adapter computation (adapter weights are float32)
        original_dtype = geo_volume_raw.dtype
        geo_volume_fp32 = geo_volume_raw.float()
        feat_fp32 = features_left[0].float()
        high_error_mask_fp32 = (
            high_error_mask.float() if high_error_mask is not None else None
        )
        
        # Apply adapter
        adapted_volume, aux = self.ttga(
            geo_volume_fp32,
            feat_fp32,
            high_error_mask=high_error_mask_fp32,
            return_aux=return_aux
        )
        
        # Convert back to original dtype
        adapted_volume = adapted_volume.to(original_dtype)
        
        return adapted_volume, aux

    def build_init_disp_and_geo_fn(self, match_left, match_right, geo_volume):
        """
        Build initial disparity and geometric function from cost volume.
        """
        # Mixed-precision safety: classifier weights are fp32, but geo_volume may be fp16 under autocast
        geo_volume_cls = geo_volume.float() if geo_volume.dtype != torch.float32 else geo_volume

        prob = F.softmax(self.classifier(geo_volume_cls).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.max_disp // 4)

        geo_fn = CombinedGeometryEncodingVolume(
            match_left.float(),
            match_right.float(),
            geo_volume_cls.float(),
            radius=self.args.corr_radius,
            num_levels=self.args.corr_levels
        )

        aux = {
            "prob": prob,
            "init_disp": init_disp,
        }

        return prob, init_disp, geo_fn

    def recurrent_refine(self, init_disp, geo_fn, net_list, inp_list, stem_2x,
                         iters=12, test_mode=False, adaptive_cfg=None,
                         matching_error_map=None, return_aux=False,
                         prediction_iters=None):
        """
        GRU-based iterative refinement of disparity.
        """
        disp = init_disp
        disp_preds = []
        aux_device = init_disp.device
        refine_aux = {
            "iter_stats": [],
            "early_stopped": torch.zeros((), device=aux_device, dtype=torch.bool),
            "early_stop_iter": torch.zeros((), device=aux_device, dtype=torch.long),
            "num_iters": torch.zeros((), device=aux_device, dtype=torch.long),
        }
        
        b, c, h, w = net_list[0].shape
        coords = torch.arange(w).float().to(disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        pair_error_mean = None
        prediction_iters = (
            None if prediction_iters is None else set(prediction_iters)
        )
        if matching_error_map is not None:
            pair_error_mean = matching_error_map.detach().mean().item()
        
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                ret = self.update_block(
                    net_list, inp_list, geo_feat, disp,
                    iter16=self.args.n_gru_layers==3,
                    iter08=self.args.n_gru_layers>=2,
                    return_delta_only_stats=return_aux
                )
                
                if return_aux:
                    net_list, mask_feat_4, delta_disp, stats = ret
                    refine_aux["iter_stats"].append(stats)
                else:
                    net_list, mask_feat_4, delta_disp = ret

            disp = disp + delta_disp
            refine_aux["num_iters"] = torch.tensor(itr + 1, device=disp.device, dtype=torch.long)
            
            # Early stopping with adaptive scheduler
            stop_now = False
            if adaptive_cfg is not None and return_aux:
                delta_mean = stats.get("delta_abs_mean", 0.0).item() if torch.is_tensor(stats.get("delta_abs_mean", 0)) else 0.0
                stop_now = adaptive_cfg.should_stop(itr, delta_mean, pair_error_mean)
                stats["pair_error_mean"] = torch.tensor(
                    0.0 if pair_error_mean is None else pair_error_mean,
                    device=delta_disp.device,
                    dtype=delta_disp.dtype,
                )
                stats["early_stop"] = torch.tensor(stop_now, device=delta_disp.device, dtype=torch.bool)
            
            if prediction_iters is not None and itr + 1 not in prediction_iters and not stop_now:
                continue
            if prediction_iters is None and test_mode and itr < iters - 1 and not stop_now:
                continue
            
            # Upsample
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

            if stop_now:
                refine_aux["early_stopped"] = torch.ones((), device=disp.device, dtype=torch.bool)
                refine_aux["early_stop_iter"] = torch.tensor(itr + 1, device=disp.device, dtype=torch.long)
                break

        # This is expressed in the recurrent low-resolution disparity units.
        # It is needed to define the warm-up update signal unambiguously.
        refine_aux["final_disp_lowres"] = disp

        return disp_up if disp_preds else disp.unsqueeze(1), disp_preds, refine_aux

    @torch.no_grad()
    def forward_features_only(self, image1, image2):
        """Return cached features without refinement."""
        return self.encode_pair(image1, image2)

    def forward_from_cache(self, cache, iters=12, test_mode=True, return_aux=False,
                           high_error_mask=None, override_geo_volume=None, adaptive_cfg=None,
                           matching_error_map_override=None, prediction_iters=None):
        """Forward pass using cached features."""
        geo_volume_raw = cache["geo_volume_raw"]
        
        if override_geo_volume is not None:
            geo_volume = override_geo_volume
            adapter_aux = {}
        else:
            geo_volume, adapter_aux = self.apply_ttga(
                geo_volume_raw,
                cache["features_left"],
                high_error_mask=high_error_mask,
                return_aux=return_aux
            )

        prob, init_disp, geo_fn = self.build_init_disp_and_geo_fn(
            cache["match_left"],
            cache["match_right"],
            geo_volume
        )
        matching_error_map = matching_error_map_override

        disp_up, disp_preds, refine_aux = self.recurrent_refine(
            init_disp=init_disp,
            geo_fn=geo_fn,
            net_list=cache["net_list"].copy(),
            inp_list=cache["inp_list"].copy(),
            stem_2x=cache["stem_2x"],
            iters=iters,
            test_mode=test_mode,
            adaptive_cfg=adaptive_cfg,
            matching_error_map=matching_error_map,
            return_aux=return_aux,
            prediction_iters=prediction_iters,
        )

        if not return_aux:
            return disp_up
        
        aux = {
            "geo_volume_raw": geo_volume_raw,
            "geo_volume": geo_volume,
            "prob": prob,
            "matching_error_map": matching_error_map,
            "init_disp_lowres": init_disp,
            "adapter_aux": adapter_aux,
            "refine_aux": refine_aux,
            "match_left": cache["match_left"],
            "match_right": cache["match_right"],
        }
        
        return disp_up, aux


    def forward(self, image1, image2, iters=12, test_mode=False,
                return_aux=False, high_error_mask=None, override_geo_volume=None,
                adaptive_cfg=None, prediction_iters=None):
        """Estimate disparity between pair of frames"""
        
        # Encode pair (static feature extraction)
        cache = self.encode_pair(image1, image2)
        
        geo_volume_raw = cache["geo_volume_raw"]
        
        # Apply adapter if available and not overridden
        if override_geo_volume is not None:
            geo_volume = override_geo_volume
            adapter_aux = {}
        else:
            geo_volume, adapter_aux = self.apply_ttga(
                geo_volume_raw,
                cache["features_left"],
                high_error_mask=high_error_mask,
                return_aux=return_aux
            )
        
        # Build initial disparity and geometric function
        prob, init_disp, geo_fn = self.build_init_disp_and_geo_fn(
            cache["match_left"],
            cache["match_right"],
            geo_volume
        )
        matching_error_map = None
        
        # Refine with GRU
        disp_up, disp_preds, refine_aux = self.recurrent_refine(
            init_disp=init_disp,
            geo_fn=geo_fn,
            net_list=cache["net_list"],
            inp_list=cache["inp_list"],
            stem_2x=cache["stem_2x"],
            iters=iters,
            test_mode=test_mode,
            adaptive_cfg=adaptive_cfg,
            matching_error_map=matching_error_map,
            return_aux=return_aux,
            prediction_iters=prediction_iters,
        )
        
        if not return_aux:
            if test_mode:
                return disp_up
            else:
                # Training mode: return initial and sequence
                with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                    xspx = self.spx_4(cache["features_left"][0])
                    xspx = self.spx_2(xspx, cache["stem_2x"])
                    spx_pred = self.spx(xspx)
                    spx_pred = F.softmax(spx_pred, 1)
                
                init_disp_up = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
                return init_disp_up, disp_preds
        else:
            aux = {
                "geo_volume_raw": geo_volume_raw,
                "geo_volume": geo_volume,
                "prob": prob,
                "matching_error_map": matching_error_map,
                "init_disp_lowres": init_disp,
                "adapter_aux": adapter_aux,
                "refine_aux": refine_aux,
                "match_left": cache["match_left"],
                "match_right": cache["match_right"],
            }
            
            if test_mode:
                return disp_up, aux
            else:
                with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                    xspx = self.spx_4(cache["features_left"][0])
                    xspx = self.spx_2(xspx, cache["stem_2x"])
                    spx_pred = self.spx(xspx)
                    spx_pred = F.softmax(spx_pred, 1)
                
                init_disp_up = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
                return init_disp_up, disp_preds, aux
