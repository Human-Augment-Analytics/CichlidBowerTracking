from typing import Tuple, Dict

import pandas as pd
import numpy as np

class TripletMiner:
    def __init__(self, num_triplets: int, diff_ratios=[0.25, 0.5, 0.25], reason='init'):
        assert sum(diff_ratios) == 0, f'Invalid Difficulty Ratios: need to sum to zero (got sum {sum(diff_ratios)})'
        assert reason in {'init', 're-mine'}, f'Invalid Reason: expected "init" or "re-mine" (got {reason})'

        self.num_triplets = num_triplets
        self.diff_ratios = diff_ratios
        self.reason = reason

    def _hard_mine(self, embeddings: Dict[int, np.ndarray], anchor: int) -> Tuple[np.ndarray, np.ndarray]:
        tmp = {identity: embedding for identity, embedding in embeddings.items() if identity != anchor}

        hard_pos, hard_neg = None, None
        # ===================================================
        # TODO: Implement hard-mining for...
        #   - positives
        #   - negatives
        # ===================================================

        del tmp
        return hard_pos, hard_neg