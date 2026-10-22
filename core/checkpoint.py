"""Checkpoint I/O for AdRe-Stereo."""

import logging

import torch


def _canonicalize_state_keys(state):
    """Translate pre-paper adapter checkpoint keys to TTGA terminology.

    Translation is confined to checkpoint I/O; public model, class, and CLI
    names remain those used in the paper.
    """
    renamed = {}
    replacements = (
        ("geo_ttt_adapter.lowrank.", "ttga.low_rank_residual."),
        ("geo_ttt_adapter.spatial.", "ttga.spatial_residual."),
        ("geo_ttt_adapter.disp_reweight.", "ttga.disparity_gate."),
        ("geo_ttt_adapter.alpha", "ttga.a_lr"),
        ("geo_ttt_adapter.beta", "ttga.a_sp"),
    )
    for key, value in state.items():
        prefix = "module." if key.startswith("module.") else ""
        body = key[len(prefix):]
        for old, new in replacements:
            if body.startswith(old):
                body = new + body[len(old):]
                break
        canonical = prefix + body
        if canonical in renamed:
            raise RuntimeError(
                f"duplicate checkpoint key after TTGA migration: {canonical}"
            )
        renamed[canonical] = value
    return renamed


def load_checkpoint(model, path, strict_ttga=False, require_low_rank_ttga=False):
    """Load plain/DataParallel checkpoints and return stored metadata."""
    payload = torch.load(path, map_location="cpu")
    wrapped = isinstance(payload, dict) and "state_dict" in payload
    metadata = payload if wrapped else {}
    state = payload["state_dict"] if wrapped else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {path!s} does not contain a state dictionary")

    state = _canonicalize_state_keys(state)
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    target_has_module = any(key.startswith("module.") for key in model.state_dict())
    if target_has_module:
        state = {f"module.{key}": value for key, value in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    ttga_marker = "module.ttga." if target_has_module else "ttga."
    missing_ttga = [key for key in missing if key.startswith(ttga_marker)]
    missing_backbone = [key for key in missing if not key.startswith(ttga_marker)]

    if missing_backbone or unexpected:
        raise RuntimeError(
            "checkpoint is incompatible with the AdRe-Stereo backbone: "
            f"missing={missing_backbone}, unexpected={unexpected}"
        )
    if strict_ttga and missing_ttga:
        raise RuntimeError(f"checkpoint is missing TTGA keys: {missing_ttga}")
    low_rank_marker = f"{ttga_marker}low_rank_residual."
    missing_low_rank = [
        key for key in missing_ttga if key.startswith(low_rank_marker)
    ]
    if require_low_rank_ttga and missing_low_rank:
        raise RuntimeError(
            "The requested checkpoint must contain the low-rank TTGA branch; "
            f"missing keys: {missing_low_rank}"
        )
    if missing_ttga:
        logging.warning(
            "Checkpoint has no values for %d TTGA parameters; identity-safe "
            "initialization is retained.",
            len(missing_ttga),
        )
    else:
        logging.info("Checkpoint loaded with all model keys matched")

    return {
        key: value for key, value in metadata.items() if key != "state_dict"
    }


def checkpoint_payload(model, **metadata):
    return {"state_dict": model.state_dict(), **metadata}
