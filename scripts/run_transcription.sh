#!/bin/zsh
# Local folder action: one transcript + one speaker-labeled transcript, no routing.
# Automator -> Folder Action -> Run Shell Script: /bin/zsh, Pass input: as arguments.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

BASE="$HOME/Documents/Meetings"
OUT="$BASE/Transcripts"
MODEL="$HOME/whisper-models/ggml-medium.bin"
PYTHON="$HOME/.venvs/diarization/bin/python"
SCRIPTS="$BASE/scripts"
ASK="$SCRIPTS/ask_speakers.applescript"
DIARIZE="$SCRIPTS/diarize.py"
NOTIFY="$SCRIPTS/notify.applescript"
LOG="$OUT/transcription.log"
PROCESS_LOCK="$BASE/.transcription.lock"
DIALOG_LOCK="$BASE/.transcription-dialog.lock"
DONE="$BASE/.processed"

mkdir -p "$OUT" "$DONE"
log() { print -r -- "$(date '+%Y-%m-%d %H:%M:%S'): $1" >> "$LOG"; }
notice() { /usr/bin/osascript "$NOTIFY" "$1" >/dev/null 2>&1 || true; }

for F in "$ASK" "$DIARIZE" "$NOTIFY" "$MODEL" "$PYTHON"; do
  if [[ ! -e "$F" ]]; then
    log "Missing required component: $F"
    exit 1
  fi
done
for TOOL in ffmpeg whisper-cli; do
  if ! command -v "$TOOL" >/dev/null; then
    log "Missing tool: $TOOL"
    exit 1
  fi
done

DIALOG_HELD=0
PROCESS_HELD=0
cleanup() {
  if (( DIALOG_HELD )); then rmdir "$DIALOG_LOCK" 2>/dev/null || true; fi
  if (( PROCESS_HELD )); then rmdir "$PROCESS_LOCK" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

# Prevent two Folder Action instances from showing popups on top of each other.
while ! mkdir "$DIALOG_LOCK" 2>/dev/null; do sleep 1; done
DIALOG_HELD=1

typeset -a INPUTS SPEAKER_COUNTS IDS
for FILE in "$@"; do
  [[ -f "$FILE" ]] || continue
  case "${FILE:l}" in
    *.m4a|*.mp3|*.wav|*.mp4|*.aac|*.flac|*.ogg) ;;
    *) continue ;;
  esac
  NAME="${FILE:t:r}"
  ID=$(printf '%s' "${FILE}:$(stat -f '%z:%m' "$FILE")" | shasum -a 256 | awk '{print $1}')
  [[ -f "$DONE/$ID.done" ]] && continue

  SPEAKERS=$(/usr/bin/osascript "$ASK" "$NAME" 2>> "$LOG")
  if (( $? != 0 )); then
    log "Speaker dialog failed for $NAME; no transcription started"
    continue
  fi
  [[ "$SPEAKERS" == "__CANCEL__" ]] && { log "Skipped: $NAME"; continue; }
  if [[ "$SPEAKERS" != auto && ! "$SPEAKERS" =~ '^[1-9][0-9]?$' ]]; then
    log "Invalid speaker count '$SPEAKERS' for $NAME"
    continue
  fi
  if [[ "$SPEAKERS" != auto ]] && (( SPEAKERS > 20 )); then
    log "Speaker count too large for $NAME: $SPEAKERS"
    continue
  fi
  INPUTS+=("$FILE")
  SPEAKER_COUNTS+=("$SPEAKERS")
  IDS+=("$ID")
  log "Queued: $NAME, speakers=$SPEAKERS"
done
rmdir "$DIALOG_LOCK" 2>/dev/null || true
DIALOG_HELD=0

(( ${#INPUTS[@]} == 0 )) && exit 0

# Only one loaded Whisper and one loaded pyannote pipeline at any given time.
while ! mkdir "$PROCESS_LOCK" 2>/dev/null; do sleep 2; done
PROCESS_HELD=1

for (( i=1; i<=${#INPUTS[@]}; i++ )); do
  FILE="${INPUTS[i]}"
  SPEAKERS="${SPEAKER_COUNTS[i]}"
  ID="${IDS[i]}"
  [[ -f "$FILE" && ! -f "$DONE/$ID.done" ]] || continue
  NAME="${FILE:t:r}"
  WORK=$(mktemp -d)
  if [[ ! -d "$WORK" ]]; then log "Cannot create temporary folder for $NAME"; continue; fi
  log "START: $NAME, speakers=$SPEAKERS"
  notice "Началась обработка: $NAME (спикеров: $SPEAKERS)"

  if ! ffmpeg -nostdin -y -loglevel error -i "$FILE" -ar 16000 -ac 1 -c:a pcm_s16le "$WORK/audio.wav" >> "$LOG" 2>&1; then
    log "FFmpeg failed: $NAME"; notice "Ошибка чтения аудио: $NAME"
    rm -rf "$WORK"; continue
  fi

  # -l auto = detect spoken language; no -tr = do not translate into English.
  if ! whisper-cli -m "$MODEL" -f "$WORK/audio.wav" -l auto -otxt -oj -of "$WORK/result" >> "$LOG" 2>&1; then
    log "Whisper failed: $NAME"; notice "Ошибка транскрибации: $NAME"
    rm -rf "$WORK"; continue
  fi
  if [[ ! -s "$WORK/result.txt" || ! -s "$WORK/result.json" ]]; then
    log "Whisper produced incomplete output: $NAME"
    rm -rf "$WORK"; continue
  fi

  # Collision-safe paired filenames, e.g. Interview.txt and Interview_diarized.txt.
  SUFFIX=1
  while true; do
    STEM="$NAME"
    if (( SUFFIX > 1 )); then STEM="${NAME}_${SUFFIX}"; fi
    LABELED="$OUT/${STEM}_diarized.txt"
    [[ ! -e "$LABELED" ]] && break
    (( SUFFIX++ ))
  done

  # Plain text is useful even if speaker detection fails.
  if "$PYTHON" "$DIARIZE" "$WORK/audio.wav" "$WORK/result.json" "$WORK/labeled.txt" "$SPEAKERS" >> "$LOG" 2>&1 && [[ -s "$WORK/labeled.txt" ]]; then
    mv "$WORK/labeled.txt" "$LABELED"
    touch "$DONE/$ID.done"
    log "DONE: $LABELED"
    notice "Готово: $NAME — транскрипт с диаризацией"
  else
    log "DIARIZATION FAILED: $NAME; no output saved"
    notice "Ошибка диаризации: $NAME"
  fi
  rm -rf "$WORK"
done
