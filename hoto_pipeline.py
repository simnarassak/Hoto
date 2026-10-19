"""
Backward-compatible entrypoint for the Hoto training pipeline.

Prefer:
    python -m hoto
    python -m hoto.pipeline

This file no longer contains the notebook body. The implementation lives in
the `hoto` package so the public app can import inference helpers without
retraining models.
"""

from hoto.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
