#!/usr/bin/env python3
"""Merge whisper.cpp transcript JSON with local pyannote Community-1 turns.

Usage: diarize.py audio.wav whisper.json output.txt auto|N
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def format_time(seconds: float) -> str:
    total = int(max(0, seconds))
    return f"{total // 3600:02d}:{(total // 60) % 60:02d}:{total % 60:02d}"


def assign_speaker(start: float, end: float, turns: list[tuple[float, float, str]]):
    """Assign a Whisper text segment to the overlapping speaker turn."""
    overlaps = defaultdict(float)
    for turn_start, turn_end, label in turns:
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > 0:
            overlaps[label] += overlap
    return max(overlaps, key=overlaps.get) if overlaps else None


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("Usage: diarize.py AUDIO.wav WHISPER.json OUTPUT.txt auto|N")
    wav_file, transcript_json, outfile = map(Path, sys.argv[1:4])
    speaker_arg = sys.argv[4]
    if speaker_arg != "auto":
        try:
            speakers = int(speaker_arg)
        except ValueError as error:
            raise SystemExit("Number of speakers must be 'auto' or 1..20") from error
        if not (1 <= speakers <= 20):
            raise SystemExit("Number of speakers must be 1..20")
    else:
        speakers = None

    started = time.perf_counter()

    from huggingface_hub import get_token
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"DIARIZATION: device={device}", flush=True)
    hf_token = get_token()
    if not hf_token:
        raise SystemExit("Missing Hugging Face token. Run 'hf auth login' inside the diarization venv.")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    )
    if pipeline is None:
        raise RuntimeError("Could not load pyannote; accept model conditions on Hugging Face")

    pipeline.to(device)
    print(f"DIARIZATION: pipeline_device={pipeline.device}, load_seconds={time.perf_counter() - started:.3f}", flush=True)
    inference_started = time.perf_counter()

    # Using in-memory audio avoids optional macOS torchaudio/torchcodec decoder issues.
    wav, rate = sf.read(str(wav_file), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(wav.T.copy())
    audio_input = {"waveform": waveform, "sample_rate": rate}
    if speakers is None:
        result = pipeline(audio_input)
    else:
        result = pipeline(audio_input, num_speakers=speakers)

    if device.type == "mps":
        torch.mps.synchronize()
    print(f"DIARIZATION: device={device}, inference_seconds={time.perf_counter() - inference_started:.3f}", flush=True)

    # Exclusive tracks simplify the alignment of transcript timestamps to one voice.
    annotation = getattr(result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = result.speaker_diarization
    turns = [
        (segment.start, segment.end, label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    # Speaker IDs remain stable across one recording, not across separate recordings.
    seen = {}
    for _, _, label in sorted(turns, key=lambda row: row[0]):
        if label not in seen:
            seen[label] = f"Спикер {len(seen) + 1}"

    whisper_data = json.loads(transcript_json.read_text(encoding="utf-8"))
    lines = []
    for segment in whisper_data.get("transcription", []):
        content = segment.get("text", "").strip()
        if not content:
            continue
        offsets = segment.get("offsets", {})
        start = float(offsets.get("from", 0)) / 1000
        end = float(offsets.get("to", 0)) / 1000
        chosen = assign_speaker(start, end, turns)
        who = seen.get(chosen, "Спикер не определён")
        lines.append(f"[{format_time(start)}] {who}: {content}")

    if not lines:
        raise RuntimeError("Whisper JSON did not contain text segments")
    outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved labeled transcript: {outfile}", flush=True)
    print(f"DIARIZATION: device={device}, total_seconds={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
