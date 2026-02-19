import gzip
import numpy as np
import torch


def preprocess_pe_file(file_path: str, max_len: int = 4_000_000) -> torch.Tensor:
    """Read a PE file (plain or gzip) and return a model-ready input tensor."""
    try:
        with gzip.open(file_path, "rb") as f:
            raw_bytes = f.read(max_len)
    except OSError:
        with open(file_path, "rb") as f:
            raw_bytes = f.read(max_len)

    x = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int16) + 1
    return torch.tensor(x, dtype=torch.long).unsqueeze(0)
