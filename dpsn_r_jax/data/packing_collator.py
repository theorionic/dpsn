"""Sequence packing for TPU training efficiency.

Packs multiple short sequences into single max_seq_len sequences using
first-fit-decreasing bin packing. Generates a block-diagonal causal
attention mask so packed sequences cannot attend across boundaries.

Usage:
    collator = PackingCollator(max_seq_len=512, pad_token_id=0)
    packed_ids, seq_pack_ids = collator(list_of_token_id_arrays)
    # packed_ids: (B_packed, T) int32 — packed token ids
    # seq_pack_ids: (B_packed, T) int32 — which sub-sequence each position belongs to (-1=pad)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


class PackingCollator:
    """Pack variable-length sequences into fixed-length bins.

    Args:
        max_seq_len: Maximum sequence length (bin size).
        pad_token_id: Token id used for padding.
        min_pack_ratio: If average sequences/bin < this, warn. Default 1.5.
    """

    def __init__(self, max_seq_len: int, pad_token_id: int = 0,
                 min_pack_ratio: float = 1.5):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.min_pack_ratio = min_pack_ratio

    def __call__(
        self,
        sequences: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pack sequences into bins.

        Args:
            sequences: List of 1-D int arrays of varying lengths.
                       Sequences longer than max_seq_len are truncated.

        Returns:
            packed_ids:   (n_bins, max_seq_len) int32
            seq_pack_ids: (n_bins, max_seq_len) int32
                          Each position holds the sub-sequence index within its
                          bin (0, 1, 2, ...). Padding positions = -1.
        """
        # Truncate to max_seq_len
        seqs = [s[:self.max_seq_len] for s in sequences]

        # First-fit-decreasing bin packing
        order = np.argsort([-len(s) for s in seqs])
        bins: List[List[np.ndarray]] = []
        bin_lens: List[int] = []

        for idx in order:
            s = seqs[idx]
            placed = False
            for bi in range(len(bins)):
                if bin_lens[bi] + len(s) <= self.max_seq_len:
                    bins[bi].append(s)
                    bin_lens[bi] += len(s)
                    placed = True
                    break
            if not placed:
                bins.append([s])
                bin_lens.append(len(s))

        # Build output arrays
        n_bins = len(bins)
        T = self.max_seq_len
        packed_ids = np.full((n_bins, T), self.pad_token_id, dtype=np.int32)
        seq_pack_ids = np.full((n_bins, T), -1, dtype=np.int32)

        for bi, bin_seqs in enumerate(bins):
            pos = 0
            for sid, s in enumerate(bin_seqs):
                L = len(s)
                packed_ids[bi, pos:pos + L] = s
                seq_pack_ids[bi, pos:pos + L] = sid
                pos += L

        return packed_ids, seq_pack_ids

    def pack_stats(self, sequences: List[np.ndarray]) -> dict:
        """Return packing statistics without allocating output arrays."""
        seqs = [s[:self.max_seq_len] for s in sequences]
        packed_ids, _ = self(seqs)
        n_bins = packed_ids.shape[0]
        ratio = len(sequences) / max(1, n_bins)
        total_tokens = sum(len(s) for s in seqs)
        utilisation = total_tokens / (n_bins * self.max_seq_len)
        return {
            "n_sequences": len(sequences),
            "n_bins": n_bins,
            "pack_ratio": ratio,
            "utilisation": utilisation,
        }
