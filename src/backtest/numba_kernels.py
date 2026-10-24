"""Small deterministic Numba kernels used by the portfolio hot path."""

import os

import numpy as np
from numba import njit


@njit(cache=True)
def top_indices_sorted_days(day_codes, ranks, original_positions, max_pos):
    """Select stable top-ranked rows from rows already ordered by day."""
    out = np.empty(len(day_codes), dtype=np.int64)
    out_count = 0
    i = 0
    while i < len(day_codes):
        day = day_codes[i]
        chosen = np.empty(max_pos, dtype=np.int64)
        chosen_ranks = np.empty(max_pos, dtype=np.float64)
        count = 0
        j = i
        while j < len(day_codes) and day_codes[j] == day:
            rank = ranks[j]
            insert = count
            if not np.isnan(rank):
                k = 0
                while k < count and (np.isnan(chosen_ranks[k]) or rank <= chosen_ranks[k]):
                    k += 1
                insert = k
            if count < max_pos:
                for k in range(count, insert, -1):
                    chosen[k] = chosen[k - 1]
                    chosen_ranks[k] = chosen_ranks[k - 1]
                chosen[insert] = original_positions[j]
                chosen_ranks[insert] = rank
                count += 1
            elif insert < max_pos:
                for k in range(max_pos - 1, insert, -1):
                    chosen[k] = chosen[k - 1]
                    chosen_ranks[k] = chosen_ranks[k - 1]
                chosen[insert] = original_positions[j]
                chosen_ranks[insert] = rank
            j += 1
        for k in range(count):
            out[out_count] = chosen[k]
            out_count += 1
        i = j
    return out[:out_count]


def numba_enabled():
    return os.getenv("NUMBA_DISABLE", os.getenv("DISABLE_NUMBA_KERNELS", "0")) not in {"1", "true", "yes"}
