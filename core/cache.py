"""Reusable stereo-pair features and raw GEV for per-pair TTGA updates."""


class StereoPairCache(dict):
    """
    Simple semantic container for caching intermediate results from encode_pair.
    Inherits from dict for easy access and serialization.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize cache as a dictionary."""
        super().__init__(*args, **kwargs)
    
    def __repr__(self):
        """Human-readable representation."""
        keys = sorted(self.keys())
        return f"StereoPairCache({', '.join(keys)})"
