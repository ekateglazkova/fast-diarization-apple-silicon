# Установка

Проверено на MacBook Air M3 (Apple Silicon), macOS 26. На Intel Mac MPS недоступен, диаризация пойдёт на CPU.

## 1. Системные инструменты

```bash
brew install ffmpeg whisper-cpp python@3.11
mkdir -p ~/whisper-models
curl -L -o ~/whisper-models/ggml-medium.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

Проверьте, что бинарник находится: `which whisper-cli`. Скрипт ищет `whisper-cli` и `ffmpeg` в `/opt/homebrew/bin` и `/usr/local/bin`.

## 2. Python-окружение

```bash
python3.11 -m venv ~/.venvs/diarization
~/.venvs/diarization/bin/pip install -r requirements.txt
```

Проверка архитектуры и GPU:

```bash
~/.venvs/diarization/bin/python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"
```

Должно быть `arm64 True`. Если `x86_64`, окружение запущено через Rosetta, и MPS работать не будет.

## 3. Доступ к модели диаризации

1. Зайдите на страницу `pyannote/speaker-diarization-community-1` на Hugging Face и примите условия.
2. Создайте read-токен в настройках Hugging Face.
3. Войдите из того же окружения:

```bash
~/.venvs/diarization/bin/hf auth login
```

Первый запуск скачает веса модели, дальше всё работает из локального кеша.

## 4. Папки и скрипты

```bash
mkdir -p ~/Documents/Meetings/{Audio,Transcripts,scripts}
cp scripts/* ~/Documents/Meetings/scripts/
chmod +x ~/Documents/Meetings/scripts/run_transcription.sh
```

Скрипт ожидает такие пути: модель `~/whisper-models/ggml-medium.bin`, Python `~/.venvs/diarization/bin/python`, папка `~/Documents/Meetings` с подпапками `Audio`, `Transcripts` и `scripts`. Если у вас другие, поправьте переменные в начале `run_transcription.sh`. Диалог выбора спикеров и уведомления вынесены в `ask_speakers.applescript` и `notify.applescript`, их тексты можно менять.

## 5. Folder Action

1. Откройте Automator → «Новый документ» → **Действие папки**.
2. Вверху выберите папку `~/Documents/Meetings/Audio`.
3. Добавьте действие **Запустить shell-скрипт**: оболочка `/bin/zsh`, «Передать ввод» → **как аргументы**.
4. Вставьте команду, которая запускает скрипт в фоне:

```bash
BASE="$HOME/Documents/Meetings"
mkdir -p "$BASE/Transcripts"
/usr/bin/nohup /bin/zsh "$BASE/scripts/run_transcription.sh" "$@" </dev/null >> "$BASE/Transcripts/transcription.log" 2>&1 &
```

Фоновый запуск через `nohup` нужен, чтобы действие Automator завершалось сразу, не дожидаясь транскрибации. Сама обработка идёт в `run_transcription.sh`, который ставит файлы в очередь.

5. Сохраните. Включите действия папки: правый клик по `Audio` → Службы → Настройка действий папки.

## 6. Проверка

Положите в `Audio` короткую запись, в окне выберите число спикеров и смотрите журнал:

```bash
tail -f ~/Documents/Meetings/Transcripts/transcription.log
```

В журнале должны появиться `START`, строки `DIARIZATION: device=mps ...` и `DONE`.

## Использование и формат вывода

Положите аудиофайл (`m4a`, `mp3`, `wav`, `mp4`, `aac`, `flac`, `ogg`) в `Audio/`. В окне укажите число спикеров: от 1 до 20 или `auto`, если оно неизвестно. «Отмена» пропускает файл.

Результат: `Transcripts/<имя>_diarized.txt`. Если такой файл уже есть, к имени добавляется `_2`, `_3` и так далее. Файл с тем же путём, размером и временем изменения обрабатывается один раз. При сбое диаризации результат не сохраняется (в журнале будет `DIARIZATION FAILED`): запустите файл ещё раз, при необходимости выбрав `auto`.

```text
[00:00:12] Спикер 1: Какую задачу мы решаем?
[00:00:19] Спикер 2: Хотим сократить время ожидания транскрипта.
```

Язык текста определяется автоматически. Подписи спикеров в скрипте русские (`Спикер N`, а `Спикер не определён`, если сегмент не пересекается ни с одним говорящим); для другого языка поменяйте строки в `scripts/diarize.py`. Пример вывода: [`examples/sample_diarized.txt`](../examples/sample_diarized.txt).

Запустить только диаризацию:

```bash
whisper-cli -m ~/whisper-models/ggml-medium.bin -f audio.wav -l auto -oj -of result
~/.venvs/diarization/bin/python scripts/diarize.py audio.wav result.json out.txt auto   # или число спикеров
```

## Если что-то не так

- **`device=cpu`** — MPS недоступен в этом окружении; проверьте шаг 2.
- **Ничего не происходит** — Folder Action не включён или скрипт не исполняемый (`chmod +x`).
- **Очередь стоит** — после принудительной остановки мог остаться каталог-блокировка. Убедитесь, что обработка не идёт (`ps aux | grep -E 'whisper|diarize'`), и удалите `~/Documents/Meetings/.transcription.lock` (и `.transcription-dialog.lock`, если окно выбора спикеров не появляется).
- **Файл не обрабатывается повторно** — успешно обработанный файл (тот же путь, размер и время изменения) пропускается. Чтобы запустить его снова, удалите маркеры из `~/Documents/Meetings/.processed`.
- **Mac уснул** — длинные записи лучше обрабатывать с подключённым питанием; на время работы можно запустить `caffeinate -i`.

## Повторить замеры

```bash
ffmpeg -i meeting.m4a -t 180 -ar 16000 -ac 1 clip.wav
PY=~/.venvs/diarization/bin/python
$PY bench/bench_diar.py clip.wav cpu auto --runs 3
$PY bench/bench_diar.py clip.wav mps auto --runs 3
$PY bench/compare_rttm.py clip.wav.cpu.run1.rttm clip.wav.mps.run1.rttm
```

`compare_rttm.py` показывает расхождение двух прогонов, а не точность относительно ручной разметки.
