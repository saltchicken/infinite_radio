#!/usr/bin/env fish

# Check if both arguments are provided using count
if test (count $argv) -lt 2
    echo "Usage: ./yt_to_sheetsage.fish <youtube-url> \"<song-name>\""
    exit 1
end

set YOUTUBE_URL $argv[1]
set SONG_NAME $argv[2]
set DATASET_DIR "/home/saltchicken/Music/music-dataset"
set COVERS_DIR (pwd)"/covers"

# Define the absolute path to the sheetsage executable in your uv environment
set SHEETSAGE_BIN (pwd)"/sheetsage/.venv/bin/sheetsage"

# Double check that the executable actually exists where we expect it
if not test -x "$SHEETSAGE_BIN"
    echo "Error: Could not find the sheetsage executable at $SHEETSAGE_BIN"
    echo "Please make sure your uv virtual environment is set up."
    exit 1
end

# Ensure both the dataset and covers directories exist
mkdir -p "$DATASET_DIR"
mkdir -p "$COVERS_DIR"

echo "Downloading and extracting audio to $DATASET_DIR as '$SONG_NAME.mp3'..."

# yt-dlp flags explained:
# --cookies-from-browser: Bypasses YouTube's bot protection
# -o "$SONG_NAME.%(ext)s": Forces the output file to use your custom name
# --exec: Runs your CLI using the absolute path to the uv environment binary
if yt-dlp -x --audio-format mp3 \
    --cookies-from-browser firefox \
    -P "$DATASET_DIR" \
    -o "$SONG_NAME.%(ext)s" \
    --exec "$SHEETSAGE_BIN \"{}\" -o \"$COVERS_DIR\"" \
    "$YOUTUBE_URL"
    
    echo "Success! Processing complete. Check the '$COVERS_DIR/$SONG_NAME-score' folder for your scores."
else
    echo "Error: yt-dlp failed to download or process the video."
    exit 1
end
