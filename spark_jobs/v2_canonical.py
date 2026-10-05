"""Spark entry for the v2 shadow cleaner.

Submit this file, not ``clean_and_embed.py``. It calls the same
``clean_partition`` implementation as the local shadow runner and restores
source order before any artifact is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_finetune.spark_v2 import main


if __name__ == "__main__":
    main()
