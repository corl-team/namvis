import torch
import torch.nn.functional as F
import numpy as np


def parse_noise_apply_strength_schedule(value):
    """Parse an optional comma-separated sequence of corruption maxima."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        parts = value.split(',')
        if any(not part.strip() for part in parts):
            raise ValueError(
                f"Invalid corruption schedule {value!r}: empty entries are not allowed."
            )
        values = tuple(float(part.strip()) for part in parts)
    else:
        values = tuple(float(item) for item in value)

    if not values:
        return None
    for scale_index, maximum in enumerate(values):
        if not np.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
            raise ValueError(
                "Corruption schedule values must be finite and in [0, 1], "
                f"but scale {scale_index} has {maximum!r}."
            )
    return values


def validate_noise_apply_strength_schedule(schedule, vae_scale_schedule):
    """Require one maximum for each scale that becomes later-scale context."""
    if schedule is None:
        return None
    expected = max(len(vae_scale_schedule) - 1, 0)
    if len(schedule) != expected:
        raise ValueError(
            "Corruption schedule length must equal len(vae_scale_schedule) - 1 "
            f"(expected {expected}, got {len(schedule)}). The final scale is excluded "
            "because it is not used as context."
        )
    return schedule


def sample_schedule_severity(context_scale_count, rng=None):
    """Sample one U(0, 1) severity per context scale."""
    if context_scale_count < 0:
        raise ValueError(f"context_scale_count must be non-negative, got {context_scale_count}.")
    sampler = np.random if rng is None else rng
    return tuple(float(value) for value in sampler.random(context_scale_count))


def corruption_rates_from_schedule(schedule, severity):
    """Convert shared scale severities into modality-specific bit-flip rates."""
    if len(schedule) != len(severity):
        raise ValueError(
            f"Corruption schedule/severity length mismatch: {len(schedule)} != {len(severity)}."
        )
    rates = []
    for scale_index, (maximum, value) in enumerate(zip(schedule, severity)):
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "Shared corruption severities must be finite and in [0, 1], "
                f"but scale {scale_index} has {value!r}."
            )
        rates.append(float(maximum) * value)
    return tuple(rates)


class BitwiseSelfCorrection(object):
    def __init__(
        self,
        vae,
        args,
        noise_apply_layers=None,
        noise_apply_requant=None,
        noise_apply_strength=None,
        view_corrupt_prob=0.0,
        view_corrupt_strength=0.4,
        view_corrupt_layers=-1,
        view_corrupt_requant=1,
        view_corrupt_min_views=2,
        noise_apply_strength_schedule=None,
    ):
        self.noise_apply_layers = args.noise_apply_layers if noise_apply_layers is None else noise_apply_layers
        self.noise_apply_requant = args.noise_apply_requant if noise_apply_requant is None else noise_apply_requant
        self.noise_apply_strength = args.noise_apply_strength if noise_apply_strength is None else noise_apply_strength
        if noise_apply_strength_schedule is None:
            noise_apply_strength_schedule = getattr(args, 'noise_apply_strength_schedule', '')
        self.noise_apply_strength_schedule = parse_noise_apply_strength_schedule(noise_apply_strength_schedule)
        self.last_noise_apply_rates = None
        self.view_corrupt_prob = view_corrupt_prob
        self.view_corrupt_strength = view_corrupt_strength
        self.view_corrupt_layers = self.noise_apply_layers if view_corrupt_layers < 0 else view_corrupt_layers
        self.view_corrupt_requant = view_corrupt_requant
        self.view_corrupt_min_views = view_corrupt_min_views
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.vae = vae

    def flip_requant(
        self,
        vae_scale_schedule,
        inp_B3HW,
        raw_features,
        device,
        view_corrupt_batch_size=None,
        view_corrupt_n_views=None,
        corruption_severity=None,
    ):
        with torch.amp.autocast('cuda', enabled = False):
            scheduled_maxima = validate_noise_apply_strength_schedule(
                self.noise_apply_strength_schedule, vae_scale_schedule
            )
            if scheduled_maxima is not None:
                if corruption_severity is None:
                    corruption_severity = sample_schedule_severity(len(scheduled_maxima))
                scheduled_rates = corruption_rates_from_schedule(
                    scheduled_maxima, corruption_severity
                )
                self.last_noise_apply_rates = tuple(scheduled_rates) + (0.0,)
            else:
                scheduled_rates = None
                self.last_noise_apply_rates = [0.0] * len(vae_scale_schedule)
            B = raw_features.shape[0]
            view_corrupt_enabled = (
                self.view_corrupt_prob > 0
                and view_corrupt_batch_size is not None
                and view_corrupt_n_views is not None
                and view_corrupt_n_views >= self.view_corrupt_min_views
                and B == view_corrupt_batch_size * view_corrupt_n_views
            )
            if view_corrupt_enabled:
                view_corrupt_sample_mask = torch.rand(view_corrupt_batch_size, device=device) < self.view_corrupt_prob
                view_corrupt_view_idx = torch.randint(view_corrupt_n_views, (view_corrupt_batch_size,), device=device)
            else:
                view_corrupt_sample_mask = None
                view_corrupt_view_idx = None
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)
            else:
                codes_out = raw_features
            cum_var_input = 0
            gt_all_bit_indices = []
            pred_all_bit_indices = []
            x_BLC_wo_prefix = []
            # print(f"vae_scale_schedule: {vae_scale_schedule}")
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                residual = codes_out - cum_var_input
                if si != len(vae_scale_schedule)-1:
                    residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=self.vae.quantizer.z_interplote_down).contiguous()
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
                gt_all_bit_indices.append(bit_indices)
                scheduled_scale = scheduled_rates is not None and si < len(scheduled_rates)
                if scheduled_scale:
                    noise_apply_strength = scheduled_rates[si]
                    mask = torch.rand(*bit_indices.shape).to(device) < noise_apply_strength
                    pred_bit_indices = bit_indices.clone()
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                elif scheduled_rates is None and si < self.noise_apply_layers:
                    noise_apply_strength = np.random.randint(0, 100 * self.noise_apply_strength+1) * 0.01
                    self.last_noise_apply_rates[si] = float(noise_apply_strength)
                    mask = torch.rand(*bit_indices.shape).to(device) < noise_apply_strength
                    pred_bit_indices = bit_indices.clone()
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                else:
                    pred_bit_indices = bit_indices

                view_corrupt_applied = view_corrupt_enabled and si < self.view_corrupt_layers and view_corrupt_sample_mask.any()
                if view_corrupt_applied:
                    pred_bit_indices = pred_bit_indices.clone()
                    pred_bits_by_view = pred_bit_indices.reshape(view_corrupt_batch_size, view_corrupt_n_views, *pred_bit_indices.shape[1:])
                    extra_strength = np.random.randint(0, 100 * self.view_corrupt_strength+1) * 0.01
                    extra_mask = torch.rand(
                        view_corrupt_batch_size,
                        view_corrupt_n_views,
                        *pred_bit_indices.shape[1:],
                        device=device,
                    ) < extra_strength
                    view_mask = (
                        torch.arange(view_corrupt_n_views, device=device)[None, :]
                        == view_corrupt_view_idx[:, None]
                    )
                    sample_mask = view_corrupt_sample_mask[:, None]
                    for _ in pred_bit_indices.shape[1:]:
                        view_mask = view_mask.unsqueeze(-1)
                        sample_mask = sample_mask.unsqueeze(-1)
                    extra_mask = extra_mask & view_mask & sample_mask
                    pred_bits_by_view[extra_mask] = 1 - pred_bits_by_view[extra_mask]
                    pred_bit_indices = pred_bits_by_view.reshape_as(pred_bit_indices)

                pred_all_bit_indices.append(pred_bit_indices)
                noise_requant = self.noise_apply_requant and (
                    scheduled_scale
                    or (scheduled_rates is None and si < self.noise_apply_layers)
                )
                if noise_requant or (self.view_corrupt_requant and view_corrupt_applied):
                    quantized = self.vae.quantizer.lfq.indices_to_codes(pred_bit_indices.float(), label_type = 'bit_label')
                cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                if si < len(vae_scale_schedule)-1:
                    this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si+1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                    if self.apply_spatial_patchify:
                        # (B,d,1,H,W) -> (B,d,H,W) -> (B,4d,H/2,W/2)
                        this_scale_input = torch.nn.functional.pixel_unshuffle(this_scale_input.squeeze(-3), 2)
                    x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)

                # print(f"this_scale_input.shape: {this_scale_input.shape}")
                # print(f"len(x_BLC_wo_prefix): {len(x_BLC_wo_prefix)}")

            if self.apply_spatial_patchify:
                gt_ms_idx_Bl = []
                for item in gt_all_bit_indices:
                    # item shape: (B,1,H,W,d)
                    item = item.squeeze(1).permute(0,3,1,2) # (B,d,H,W)
                    # (B,d,H,W) -> (B,4d,H/2,W/2)
                    item = torch.nn.functional.pixel_unshuffle(item, 2)
                    # (B,4d,H/2,W/2) -> (B,H/2,W/2,4d) -> (B,H/2*w/2,4d)
                    item = item.permute(0,2,3,1).reshape(B, -1, 4*self.vae.codebook_dim)
                    gt_ms_idx_Bl.append(item)
            else:
                gt_ms_idx_Bl = [item.reshape(B, -1, self.vae.codebook_dim) for item in gt_all_bit_indices]
            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)


        return x_BLC_wo_prefix, gt_ms_idx_Bl

    def flip_requant_nonoise(self, vae_scale_schedule, inp_B3HW, raw_features, device):
        with torch.amp.autocast('cuda', enabled = False):
            B = raw_features.shape[0]
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)
            else:
                codes_out = raw_features
            cum_var_input = 0
            gt_all_bit_indices = []
            pred_all_bit_indices = []
            x_BLC_wo_prefix = []
            # print(f"vae_scale_schedule: {vae_scale_schedule}")
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                residual = codes_out - cum_var_input
                if si != len(vae_scale_schedule)-1:
                    residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=self.vae.quantizer.z_interplote_down).contiguous()
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
                gt_all_bit_indices.append(bit_indices)
                # if si < self.noise_apply_layers:
                #     noise_apply_strength = np.random.randint(0, 100 * self.noise_apply_strength+1) * 0.01
                #     mask = torch.rand(*bit_indices.shape).to(device) < noise_apply_strength
                #     pred_bit_indices = bit_indices.clone()
                #     pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                #     pred_all_bit_indices.append(pred_bit_indices)
                #     if self.noise_apply_requant:
                #         quantized = self.vae.quantizer.lfq.indices_to_codes(pred_bit_indices, label_type = 'bit_label')
                # else:
                #     pred_all_bit_indices.append(bit_indices)
                cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                if si < len(vae_scale_schedule):
                    this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si], mode=self.vae.quantizer.z_interplote_up).contiguous()
                    if self.apply_spatial_patchify:
                        # (B,d,1,H,W) -> (B,d,H,W) -> (B,4d,H/2,W/2)
                        this_scale_input = torch.nn.functional.pixel_unshuffle(this_scale_input.squeeze(-3), 2)
                    x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)

                # print(f"this_scale_input.shape: {this_scale_input.shape}")
                # print(f"len(x_BLC_wo_prefix): {len(x_BLC_wo_prefix)}")

            if self.apply_spatial_patchify:
                gt_ms_idx_Bl = []
                for item in gt_all_bit_indices:
                    # item shape: (B,1,H,W,d)
                    item = item.squeeze(1).permute(0,3,1,2) # (B,d,H,W)
                    # (B,d,H,W) -> (B,4d,H/2,W/2)
                    item = torch.nn.functional.pixel_unshuffle(item, 2)
                    # (B,4d,H/2,W/2) -> (B,H/2,W/2,4d) -> (B,H/2*w/2,4d)
                    item = item.permute(0,2,3,1).reshape(B, -1, 4*self.vae.codebook_dim)
                    gt_ms_idx_Bl.append(item)
            else:
                gt_ms_idx_Bl = [item.reshape(B, -1, self.vae.codebook_dim) for item in gt_all_bit_indices]
            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)

        
        return x_BLC_wo_prefix, gt_ms_idx_Bl
