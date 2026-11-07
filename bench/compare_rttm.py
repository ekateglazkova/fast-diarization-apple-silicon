#!/usr/bin/env python3
"""Diarization error rate between two RTTM files (e.g. CPU vs MPS output).

Usage: python bench/compare_rttm.py reference.rttm hypothesis.rttm

This measures agreement between two runs, not accuracy against human labels.
"""
import sys

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate


def first(path):
    return next(iter(load_rttm(path).values()))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: compare_rttm.py reference.rttm hypothesis.rttm")
    der = DiarizationErrorRate()(first(sys.argv[1]), first(sys.argv[2]))
    print(f"DER between runs: {der:.4f}")
