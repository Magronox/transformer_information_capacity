"""Min-d search with the OLMo-3 transformer."""
from __future__ import annotations

import functools
import sys

import search_min_dim
import train
from olmo3.model import build_olmo3

DEFAULT_DIRS = ["--db-dir", "studies_olmo3", "--cache-dir", "probe_cache_olmo3"]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    search_min_dim.main(DEFAULT_DIRS + argv,
                        train_fn=functools.partial(train.train, build_model=build_olmo3))


if __name__ == "__main__":
    main()
