#!/usr/bin/env fish

if test (count $argv) -lt 1
    echo "Usage: ./batch_sheetsage.fish <queue.csv>"
    exit 1
end

set BATCH_FILE $argv[1]

if not test -f "$BATCH_FILE"
    echo "Error: File '$BATCH_FILE' not found."
    exit 1
end

# Create a temporary file to hold our transcription tasks
set TASK_LIST (mktemp)

# Phase 1: Download all files
cat $BATCH_FILE | while read -l line
    if test -z "$line"; or string match -q -r '^\s*#' -- "$line"
        continue
    end

    # Changed delimiter from "|" to "," to support CSV format
    set args (string split "," -- $line)
    set YOUTUBE_URL (string trim -- $args[1])
    set SONG_NAME (string trim -- $args[2])
    set SUBFOLDER (string trim -- $args[3])

    set DATASET_DIR /home/saltchicken/Music/music-dataset
    set COVERS_DIR (pwd)"/covers"

    if test -n "$SUBFOLDER"
        set DATASET_DIR "$DATASET_DIR/$SUBFOLDER"
        set COVERS_DIR "$COVERS_DIR/$SUBFOLDER"
    end

    mkdir -p "$DATASET_DIR"
    mkdir -p "$COVERS_DIR"

    echo "=================================================="
    echo "⬇️  Downloading: $SONG_NAME"
    echo "=================================================="

    # Run yt-dlp WITHOUT the --exec command
    yt-dlp -x --audio-format mp3 \
        --cookies-from-browser firefox \
        -P "$DATASET_DIR" \
        -o "$SONG_NAME.%(ext)s" \
        "$YOUTUBE_URL"

    # Add the expected file paths to the task list (keep | here as sheetsage parses it)
    set AUDIO_FILE "$DATASET_DIR/$SONG_NAME.mp3"
    echo "$AUDIO_FILE|$COVERS_DIR" >>$TASK_LIST
end

# Phase 2: Transcribe all files with the model loaded once
echo "=================================================="
echo "🧠 Loading Model & Starting Batch Transcription..."
echo "=================================================="

set SHEETSAGE_BIN (pwd)"/sheetsage/.venv/bin/sheetsage"

if not test -x "$SHEETSAGE_BIN"
    echo "Error: Could not find the sheetsage executable at $SHEETSAGE_BIN"
    rm $TASK_LIST
    exit 1
end

# Feed the generated list to the python script
$SHEETSAGE_BIN --batch $TASK_LIST

# Cleanup
rm $TASK_LIST
echo "✅ Entire batch operation complete!"
