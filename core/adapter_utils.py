"""TTGA parameter and state helpers."""

import copy


def get_ttga_parameters(model):
    """Return the trainable parameters belonging to TTGA."""
    if not hasattr(model, "ttga") or model.ttga is None:
        return []

    return [p for p in model.ttga.parameters() if p.requires_grad]


def clone_ttga_state(model):
    """Create a deep copy of TTGA state for per-pair reset."""
    if not hasattr(model, "ttga") or model.ttga is None:
        return None

    return copy.deepcopy(model.ttga.state_dict())


def restore_ttga_state(model, state_dict):
    """Restore TTGA to its saved per-pair initialization."""
    if state_dict is None or not hasattr(model, "ttga") or model.ttga is None:
        return

    model.ttga.load_state_dict(state_dict, strict=True)
