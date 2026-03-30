import numpy as np


class MixedDataset:
    """
    Serves batches from multiple HuggingFace datasets according to user-defined ratios.

    On each get_batch() call, samples which dataset to pull from using numpy random
    choice with probabilities = normalized ratios.

    Args:
        datasets: list of ChunkedHFDataset instances
        ratios: list of floats (will be normalized to sum=1)
        seed: random seed for reproducibility
    """

    def __init__(self, datasets, ratios, seed=42):
        assert len(datasets) == len(ratios), "datasets and ratios must have the same length"
        assert len(datasets) > 0, "must provide at least one dataset"

        self.datasets = datasets
        total = sum(ratios)
        self.ratios = [r / total for r in ratios]
        self.rng = np.random.default_rng(seed)
        self._batch_counts = [0] * len(datasets)
        self._total_batches = 0

        names = []
        for ds, ratio in zip(datasets, self.ratios):
            name = getattr(ds, "dataset_name", f"dataset_{len(names)}")
            short = name.split("/")[-1] if "/" in name else name
            names.append(f"{short}({ratio*100:.0f}%)")
        print(f"[MixedDataset] {len(datasets)} datasets: {', '.join(names)}")

    def get_batch(self, batch_size=None):
        while True:
            idx = int(self.rng.choice(len(self.datasets), p=self.ratios))
            try:
                batch = self.datasets[idx].get_batch(batch_size)
                self._batch_counts[idx] += 1
                self._total_batches += 1
                return batch
            except StopIteration:
                # Retry with a different dataset draw
                continue

    def get_state(self):
        return {
            "batch_counts": list(self._batch_counts),
            "total_batches": self._total_batches,
        }
