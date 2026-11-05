"""Shared utilities: random seeds, directory management, JSON serialization, moving average, and CSV logger.
Do not modify predefined seed constants.
"""
import csv
import json
import os
import random

import numpy as np

TRAIN_SEEDS = [0, 1, 2, 3, 4]
SELECT_SEED_START = 1000
TEST_SEED_START = 100
ANALYSIS_SEED_START = 2000
DEMO_SEED_START = 3000
DEFAULT_LOG_DIR = "private/logs"
DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_WEB_INDEX = "public/web/index.html"


def set_seed(seed: int) -> None:
    """Set global seed for random, numpy, and torch.

    Input:  seed -- integer random seed.
    Output: None (mutates global state).
    Flow:   call random.seed, np.random.seed, and torch.manual_seed in order.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> None:
    """Create directory and parent directories if they do not exist.

    Input:  path -- destination directory path string.
    Output: None.
    Flow:   if path is non-empty, call os.makedirs with exist_ok=True.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    """Serialize a Python object to a formatted UTF-8 JSON file.

    Input:  obj  -- serializable object;
            path -- target output file path string.
    Output: None (writes to disk).
    Flow:   ensure parent directory exists, open file, call json.dump with indent=2.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def trailing_ma(x, w: int = 100):
    """Compute trailing moving average where element i is average of up to w recent elements ending at i.

    Input:  x -- 1D float sequence (list or ndarray);
            w -- moving window size (default 100).
    Output: ndarray of floats matching input length.
    Flow:   use cumulative sum to calculate rolling average in O(n) without edge padding.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    c = np.cumsum(x)
    idx = np.arange(len(x))
    lo = idx - w
    prev = np.where(lo >= 0, c[np.maximum(lo, 0)], 0.0)
    cnt = np.minimum(idx + 1, w)
    return (c - prev) / cnt


class CsvLogger:
    """CSV logger that flushes after every write to preserve logs during abrupt terminations.

    Input (init): path   -- CSV file destination path;
                  fields -- ordered sequence of column header names.
    Output: formatted CSV file on disk.
    Flow:   open file, write header row immediately; each log() writes a row and flushes buffer.
    """

    def __init__(self, path: str, fields):
        ensure_dir(os.path.dirname(path))
        self.fields = list(fields)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)
        self.w.writeheader()

    def log(self, row: dict) -> None:
        """Write a dictionary row to the CSV file and flush to disk.

        Input:  row -- dict mapping column names to values; missing columns filled with empty string.
        Output: None (writes to disk).
        Flow:   extract values for self.fields, write row via DictWriter, flush file buffer.
        """
        self.w.writerow({k: row.get(k, "") for k in self.fields})
        self.f.flush()

    def close(self) -> None:
        """Close the underlying CSV file handle.

        Input:  None.
        Output: None.
        Flow:   invoke self.f.close().
        """
        self.f.close()
