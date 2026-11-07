# Setup

[По-русски](setup.md) · [← Back to the README](../README.md)

Tested on a MacBook Air M3 (Apple Silicon), macOS 26. On an Intel Mac MPS is not available and diarization runs on the CPU.

## 1. System tools

```bash
brew install ffmpeg whisper-cpp python@3.11
mkdir -p ~/whisper-models
curl -L -o ~/whisper-models/ggml-medium.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

Check that the binary is found: `which whisper-cli`. The script looks for `whisper-cli` and `ffmpeg` in `/opt/homebrew/bin` and `/usr/local/bin`.

## 2. Python environment

```bash
python3.11 -m venv ~/.venvs/diarization
~/.venvs/diarization/bin/pip install -r requirements.txt
```

Check the architecture and the GPU:

```bash
~/.venvs/diarization/bin/python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"
```

It must print `arm64 True`. If it prints `x86_64`, the environment runs under Rosetta and MPS will not work.

## 3. Access to the diarization model

1. Open the `pyannote/speaker-diarization-community-1` page on Hugging Face and accept the conditions.
2. Create a read token in your Hugging Face settings.
3. Log in from the same environment:

```bash
~/.venvs/diarization/bin/hf auth login
```

The first run downloads the model weights; after that everything works from the local cache.

## 4. Folders and scripts

```bash
mkdir -p ~/Documents/Meetings/{Audio,Transcripts,scripts}
cp scripts/* ~/Documents/Meetings/scripts/
chmod +x ~/Documents/Meetings/scripts/run_transcription.sh
```

The script expects the model at `~/whisper-models/ggml-medium.bin`, Python at `~/.venvs/diarization/bin/python`, and the folder `~/Documents/Meetings` with the subfolders `Audio`, `Transcripts` and `scripts`. If yours differ, edit the variables at the top of `run_transcription.sh`. The speaker dialog and the notifications live in `ask_speakers.applescript` and `notify.applescript`; you can change their texts.

## 5. Folder Action

1. Open Automator → New Document → **Folder Action**.
2. At the top, choose the folder `~/Documents/Meetings/Audio`.
3. Add the **Run Shell Script** action: shell `/bin/zsh`, pass input **as arguments**.
4. Paste the command that starts the script in the background:

```bash
BASE="$HOME/Documents/Meetings"
mkdir -p "$BASE/Transcripts"
/usr/bin/nohup /bin/zsh "$BASE/scripts/run_transcription.sh" "$@" </dev/null >> "$BASE/Transcripts/transcription.log" 2>&1 &
```

Starting it in the background with `nohup` lets the Automator action finish at once, without waiting for transcription. The processing itself runs in `run_transcription.sh`, which puts files in a queue.

5. Save. Enable folder actions: right-click `Audio` → Services → Folder Actions Setup.

## 6. Check that it works

Drop a short recording into `Audio`, choose the number of speakers in the dialog and watch the log:

```bash
tail -f ~/Documents/Meetings/Transcripts/transcription.log
```

The log should show `START`, the `DIARIZATION: device=mps ...` lines and `DONE`.

## Usage and output format

Drop an audio file (`m4a`, `mp3`, `wav`, `mp4`, `aac`, `flac`, `ogg`) into `Audio/`. A dialog asks for the number of speakers: a number from 1 to 20, or `auto` if you do not know. Cancel skips the file.

The result is `Transcripts/<name>_diarized.txt`. If such a file already exists, `_2`, `_3` and so on are added to the name. A file with the same path, size and modification time is processed only once. If diarization fails, nothing is saved (the log shows `DIARIZATION FAILED`): run the file again, with `auto` if needed.

```text
[00:00:12] Спикер 1: Какую задачу мы решаем?
[00:00:19] Спикер 2: Хотим сократить время ожидания транскрипта.
```

The text is in whatever language Whisper detected; the speaker labels in the script are Russian (`Спикер N`, and `Спикер не определён` when a segment has no overlap with any speaker turn). Change the strings in `scripts/diarize.py` for another language. A synthetic example is in [`examples/sample_diarized.txt`](../examples/sample_diarized.txt).

To run the diarization step on its own:

```bash
whisper-cli -m ~/whisper-models/ggml-medium.bin -f audio.wav -l auto -oj -of result
~/.venvs/diarization/bin/python scripts/diarize.py audio.wav result.json out.txt auto   # or a number of speakers
```

## If something is wrong

- **`device=cpu`**: MPS is not available in this environment; check step 2.
- **Nothing happens**: the Folder Action is not enabled, or the script is not executable (`chmod +x`).
- **The queue is stuck**: after a force quit a lock directory may be left behind. Make sure nothing is running (`ps aux | grep -E 'whisper|diarize'`) and delete `~/Documents/Meetings/.transcription.lock` (and `.transcription-dialog.lock` if the speaker dialog does not appear).
- **A file is not processed again**: a file that was processed successfully (same path, size and modification time) is skipped. To run it again, delete the markers from `~/Documents/Meetings/.processed`.
- **The Mac fell asleep**: process long recordings with the power plugged in; you can run `caffeinate -i` while it works.

## Repeat the measurements

```bash
ffmpeg -i meeting.m4a -t 180 -ar 16000 -ac 1 clip.wav
PY=~/.venvs/diarization/bin/python
$PY bench/bench_diar.py clip.wav cpu auto --runs 3
$PY bench/bench_diar.py clip.wav mps auto --runs 3
$PY bench/compare_rttm.py clip.wav.cpu.run1.rttm clip.wav.mps.run1.rttm
```

`compare_rttm.py` shows how much two runs disagree, not accuracy against a human reference.
