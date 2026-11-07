#!/usr/bin/env python3
"""Compare pyannote community-1 diarization on CPU vs MPS on the same audio.

Usage:
  python bench/bench_diar.py clip.wav cpu auto
  python bench/bench_diar.py clip.wav mps auto
  python bench/bench_diar.py clip.wav mps 2 --runs 3

Writes an RTTM per run, so the outputs can be compared, and prints a JSON line
with timings that you can collect into a table.
"""
import argparse
import json
import os
import resource
import statistics
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import get_token  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("device", choices=["cpu", "mps"])
    p.add_argument("speakers", help="auto or a number")
    p.add_argument("--runs", type=int, default=1)
    a = p.parse_args()

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=get_token())
    pipe.to(torch.device(a.device))

    w, sr = sf.read(a.wav, dtype="float32", always_2d=True)
    audio = {"waveform": torch.from_numpy(w.T.copy()), "sample_rate": sr}
    duration = len(w) / sr
    kwargs = {} if a.speakers == "auto" else {"num_speakers": int(a.speakers)}

    times = []
    for i in range(a.runs):
        c0 = time.process_time()
        t0 = time.perf_counter()
        out = pipe(audio, **kwargs)
        if a.device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        cpu = time.process_time() - c0
        times.append(dt)
        name = f"{os.path.basename(a.wav)}.{a.device}.run{i + 1}.rttm"
        with open(name, "w") as f:
            out.exclusive_speaker_diarization.write_rttm(f)
        print(json.dumps({
            "device": a.device, "run": i + 1, "seconds": round(dt, 3),
            "audio_seconds": round(duration, 2), "rtf": round(dt / duration, 3),
            "avg_cpu_pct": round(100 * cpu / dt, 1), "rttm": name,
        }, ensure_ascii=False))

    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(json.dumps({
        "device": a.device, "runs": a.runs,
        "median_seconds": round(statistics.median(times), 3),
        "peak_rss_gib": round(peak_gib, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
