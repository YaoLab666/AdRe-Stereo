"""Pair-level adaptive recurrent refinement from AdRe-Stereo."""

class PairLevelAdaptiveRefinement:
    """
    Stop when update magnitude and pair-level matching error are both small.
    """
    
    def __init__(self, delta_thresh=0.02, pair_error_thresh=0.25,
                 min_iters=10, max_iters=32):
        """
        Args:
            delta_thresh: stop if the mean absolute recurrent disparity update
                is below this value. Units are the recurrent low-resolution
                disparity grid (quarter resolution when n_downsample=2).
            pair_error_thresh: stop if mean pair-level matching error is below this value
            min_iters: minimum iterations before early-exit
            max_iters: maximum allowed iterations
        """
        self.delta_thresh = delta_thresh
        self.pair_error_thresh = pair_error_thresh
        self.min_iters = min_iters
        self.max_iters = max_iters
        
    def should_stop(self, itr, delta_abs_mean, pair_error_mean=None):
        """
        Determine if iteration should stop.
        
        Args:
            itr: current iteration number (0-indexed)
            delta_abs_mean: mean absolute disparity change
            pair_error_mean: mean pair-level matching error (optional)
            
        Returns:
            should_stop: boolean
        """
        # Always run minimum iterations
        if itr + 1 < self.min_iters:
            return False
        
        # Stop if both convergence criteria met
        converged = delta_abs_mean < self.delta_thresh
        
        if pair_error_mean is not None:
            low_error = pair_error_mean < self.pair_error_thresh
            if converged and low_error:
                return True
        elif converged:
            return True
        
        return False
    
    def get_max_iters(self):
        """Get maximum allowed iterations."""
        return self.max_iters


class FixedIterationRefinement:
    """
    Dummy scheduler that never stops early.
    Used when adaptive iteration is disabled.
    """
    
    def __init__(self, max_iters=8):
        self.max_iters = max_iters
        
    def should_stop(self, itr, delta_abs_mean, pair_error_mean=None):
        return False
    
    def get_max_iters(self):
        return self.max_iters
