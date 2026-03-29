"""Pool coverage tracking for DPSN model training.

Monitors which regions of the pool are accessed during training to ensure
efficient utilization of the 1.07B pool parameters.
"""

import json
from typing import Set, Tuple, Dict, Any, Optional
import numpy as np
import jax.numpy as jnp
from pathlib import Path


class PoolCoverageTracker:
    """Track which pool regions and vectors are accessed during training.

    Monitors:
    - 2D grid coordinates (mu_r, mu_c) accessed by the indexer
    - Actual vectors retrieved (considering retrieval window)
    - Access frequency heatmap
    """

    def __init__(self, pool_grid_rows: int, pool_grid_cols: int, window_size: int = 4):
        """Initialize pool coverage tracker.

        Args:
            pool_grid_rows: Number of rows in pool 2D grid
            pool_grid_cols: Number of columns in pool 2D grid
            window_size: Retrieval window size (e.g., 4 = 4×4 = 16 vectors per access)
        """
        self.grid_rows = pool_grid_rows
        self.grid_cols = pool_grid_cols
        self.window_size = window_size

        # Track unique 2D coordinates accessed (sparse set)
        self.accessed_coordinates: Set[Tuple[int, int]] = set()

        # Track unique vector indices accessed
        self.accessed_indices: Set[int] = set()

        # Track access frequency per coordinate (for heatmap)
        self.access_frequency: Dict[Tuple[int, int], int] = {}

        # Statistics
        self.total_accesses = 0
        self.total_vectors_accessed_count = 0

    def record_access(self, mu_r: np.ndarray, mu_c: np.ndarray,
                     window_size: Optional[int] = None) -> None:
        """Record pool accesses from one reasoning step.

        Args:
            mu_r: (B, H) or flattened array — row coordinates from indexer
            mu_c: (B, H) or flattened array — col coordinates from indexer
            window_size: Optional override of default window size
        """
        if window_size is None:
            window_size = self.window_size

        # Convert to numpy if JAX array
        if hasattr(mu_r, '__array__'):
            mu_r = np.asarray(mu_r)
        if hasattr(mu_c, '__array__'):
            mu_c = np.asarray(mu_c)

        # Flatten to 1D
        mu_r_flat = np.atleast_1d(mu_r).flatten()
        mu_c_flat = np.atleast_1d(mu_c).flatten()

        # Process each access
        for r_norm, c_norm in zip(mu_r_flat, mu_c_flat):
            try:
                # Denormalize from [0, 1] to grid coordinates
                r_float = float(r_norm) * (self.grid_rows - 1)
                c_float = float(c_norm) * (self.grid_cols - 1)

                # Round to nearest integer
                r_int = int(np.round(r_float)) % self.grid_rows
                c_int = int(np.round(c_float)) % self.grid_cols

                # Record center coordinate
                coord = (r_int, c_int)
                self.accessed_coordinates.add(coord)
                self.access_frequency[coord] = self.access_frequency.get(coord, 0) + 1

                # Record vectors in the retrieval window
                w_half = window_size // 2
                for dr in range(-w_half, w_half + 1):
                    for dc in range(-w_half, w_half + 1):
                        r_win = (r_int + dr) % self.grid_rows
                        c_win = (c_int + dc) % self.grid_cols
                        flat_idx = r_win * self.grid_cols + c_win
                        self.accessed_indices.add(flat_idx)

                self.total_accesses += 1

            except (ValueError, TypeError) as e:
                # Skip invalid values
                continue

    def get_coverage(self) -> Dict[str, Any]:
        """Get comprehensive coverage statistics.

        Returns:
            Dictionary with coverage metrics
        """
        total_coords = self.grid_rows * self.grid_cols

        coord_coverage = len(self.accessed_coordinates) / total_coords * 100 if total_coords > 0 else 0
        vector_coverage = len(self.accessed_indices) / total_coords * 100 if total_coords > 0 else 0

        # Calculate access concentration (std of frequency distribution)
        if self.access_frequency:
            frequencies = np.array(list(self.access_frequency.values()))
            access_mean = frequencies.mean()
            access_std = frequencies.std()
            access_concentration = access_std / (access_mean + 1e-8)  # Gini-like measure
        else:
            access_concentration = 0.0

        return {
            "unique_coordinates": len(self.accessed_coordinates),
            "total_coordinates": total_coords,
            "coordinate_coverage_pct": float(coord_coverage),
            "unique_vectors": len(self.accessed_indices),
            "total_vectors": total_coords,
            "vector_coverage_pct": float(vector_coverage),
            "total_accesses": int(self.total_accesses),
            "access_concentration": float(access_concentration),  # 0=uniform, high=concentrated
        }

    def get_heatmap(self) -> np.ndarray:
        """Get access frequency heatmap.

        Returns:
            (grid_rows, grid_cols) array with access counts per coordinate
        """
        heatmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for (r, c), count in self.access_frequency.items():
            if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                heatmap[r, c] = count

        return heatmap

    def get_summary_string(self) -> str:
        """Get formatted summary string for logging.

        Returns:
            Pretty-printed coverage summary
        """
        stats = self.get_coverage()

        summary = (
            f"Pool Coverage Summary:\n"
            f"  Coordinates: {stats['unique_coordinates']:,}/{stats['total_coordinates']:,} "
            f"({stats['coordinate_coverage_pct']:.1f}%)\n"
            f"  Vectors:     {stats['unique_vectors']:,}/{stats['total_vectors']:,} "
            f"({stats['vector_coverage_pct']:.1f}%)\n"
            f"  Accesses:    {stats['total_accesses']:,}\n"
            f"  Concentration: {stats['access_concentration']:.3f} "
            f"(0=uniform, high=concentrated)"
        )
        return summary

    def reset(self) -> None:
        """Reset all tracking for next epoch."""
        self.accessed_coordinates.clear()
        self.accessed_indices.clear()
        self.access_frequency.clear()
        self.total_accesses = 0
        self.total_vectors_accessed_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert coverage data to dictionary for checkpoint saving."""
        return {
            "window_size": int(self.window_size),
            "total_accesses": int(self.total_accesses),
            "access_frequency": {
                f"{r},{c}": int(count) for (r, c), count in self.access_frequency.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], grid_rows: int, grid_cols: int) -> "PoolCoverageTracker":
        """Restore coverage tracker from saved dictionary."""
        window_size = data.get("window_size", 4)
        tracker = cls(grid_rows, grid_cols, window_size=window_size)

        freq = data.get("access_frequency", {})
        for key, count in freq.items():
            r, c = int(key.split(",")[0]), int(key.split(",")[1])
            coord = (r, c)
            tracker.access_frequency[coord] = count
            tracker.accessed_coordinates.add(coord)
            # Re-expand the retrieval window so accessed_indices is fully restored
            w_half = window_size // 2
            for dr in range(-w_half, w_half + 1):
                for dc in range(-w_half, w_half + 1):
                    r_win = (r + dr) % grid_rows
                    c_win = (c + dc) % grid_cols
                    tracker.accessed_indices.add(r_win * grid_cols + c_win)

        tracker.total_accesses = data.get("total_accesses", sum(freq.values()))
        return tracker


def print_coverage_report(coverage_tracker: PoolCoverageTracker,
                         step: int,
                         title: str = "Pool Coverage") -> None:
    """Print a detailed coverage report.

    Args:
        coverage_tracker: PoolCoverageTracker instance
        step: Training step number
        title: Title for the report
    """
    stats = coverage_tracker.get_coverage()

    print("\n" + "="*60)
    print(f" {title} (Step {step})")
    print("="*60)
    print(f"Grid Dimensions:     {coverage_tracker.grid_rows}×{coverage_tracker.grid_cols}")
    print(f"\nCoordinate Coverage: {stats['unique_coordinates']:>8,}/{stats['total_coordinates']:>8,} "
          f"({stats['coordinate_coverage_pct']:>5.1f}%)")
    print(f"Vector Coverage:     {stats['unique_vectors']:>8,}/{stats['total_vectors']:>8,} "
          f"({stats['vector_coverage_pct']:>5.1f}%)")
    print(f"\nTotal Accesses:      {stats['total_accesses']:>8,}")
    print(f"Access Concentration:{stats['access_concentration']:>8.3f} "
          f"{'⚠️  CONCENTRATED' if stats['access_concentration'] > 2.0 else '✓ DISPERSED'}")

    # Assessment
    coverage_pct = stats['coordinate_coverage_pct']
    if coverage_pct >= 80:
        assessment = "✓ Excellent - Pool well-utilized"
    elif coverage_pct >= 60:
        assessment = "✓ Good - Reasonable pool coverage"
    elif coverage_pct >= 40:
        assessment = "⚠️  Moderate - Could improve coverage"
    else:
        assessment = "❌ Poor - Indexer may be stuck"

    print(f"\nAssessment: {assessment}")
    print("="*60 + "\n")


def save_coverage_report(coverage_tracker: PoolCoverageTracker,
                        checkpoint_dir: str,
                        step: int) -> None:
    """Save coverage report to checkpoint directory.

    Args:
        coverage_tracker: PoolCoverageTracker instance
        checkpoint_dir: Directory to save report
        step: Training step number
    """
    report_path = Path(checkpoint_dir) / f"pool_coverage_step_{step}.json"

    report = {
        "step": int(step),
        "coverage_data": coverage_tracker.to_dict(),
        "summary": coverage_tracker.get_coverage(),
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Saved coverage report to {report_path}")
