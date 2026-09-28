#!/bin/bash
# Download the Kinetics-400 validation videos and annotations into data/kinetics400.
#   scripts/k400val.sh [--stream]
# --stream downloads, extracts and deletes one tar at a time to save disk space.

SCRIPT_PATH="$(readlink -f "$0")"
ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"}
cd "$ROOT"

VIDEO_DIR="data/kinetics400/videos"
CSV_DIR="data/kinetics400/csv"
PATHS_URL="https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt"
PATHS_FILE="k400_val_path.txt"

STREAM_MODE=false
if [ "$1" = "--stream" ]; then
    STREAM_MODE=true
fi

mkdir -p "$VIDEO_DIR"
mkdir -p "$CSV_DIR"

echo "Downloading paths file..."
wget "$PATHS_URL" -O "$PATHS_FILE"

if $STREAM_MODE; then
    echo "[stream mode] download -> extract -> delete, processing one-by-one..."
    while read one; do
        filename=${one##*/}
        if [ -f "$filename" ]; then
            echo "Skipping download, tar already exists: $filename"
        else
            echo "Downloading: $filename"
            wget "$one" -O "$filename"
        fi
        echo "Extracting: $filename"
        tar zxf "$filename" -C "$VIDEO_DIR"
        echo "Removing tar: $filename"
        rm "$filename"
    done < "$PATHS_FILE"
else
    echo "[batch mode] downloading all tars first, then extracting..."
    while read one; do
        filename=${one##*/}
        if [ -f "$filename" ]; then
            echo "Skipping download, tar already exists: $filename"
        else
            echo "Downloading: $one"
            wget "$one"
        fi
    done < "$PATHS_FILE"

    while read one; do
        filename=${one##*/}
        tar zxf "$filename" -C "$VIDEO_DIR"
    done < "$PATHS_FILE"

    echo "Removing all tar files..."
    while read one; do
        filename=${one##*/}
        rm -f "$filename"
    done < "$PATHS_FILE"
fi

rm "$PATHS_FILE"

echo "Downloading val.csv..."
wget https://s3.amazonaws.com/kinetics/400/annotations/val.csv -P "$CSV_DIR"

echo "Done."
