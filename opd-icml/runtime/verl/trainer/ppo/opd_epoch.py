"""Complete an OPD epoch without changing the sampler's data source."""


def complete_epoch_padding(dataset_size, batch_size, expected_rows=None):
    """Validate the filtered source and return the final-batch repeat count."""
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)
    if expected_rows is not None and dataset_size != int(expected_rows):
        raise ValueError(
            f"filtered training dataset has {dataset_size} rows, expected {expected_rows}; "
            "refusing to run a misaligned full-epoch comparison"
        )
    if batch_size < 1 or dataset_size < batch_size:
        raise ValueError("full-epoch OPD requires at least one complete training batch")
    return (-dataset_size) % batch_size


class PadLastBatchCollator:
    """Repeat only rows in the final batch; keep the original sampler unchanged."""

    def __init__(self, base_collate, batch_size):
        self.base_collate = base_collate
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def __call__(self, rows):
        rows = list(rows)
        if not rows or len(rows) > self.batch_size:
            raise ValueError(f"invalid OPD batch size: {len(rows)}")
        original_size = len(rows)
        if original_size < self.batch_size:
            rows.extend(rows[index % original_size] for index in range(self.batch_size - original_size))
        return self.base_collate(rows)
