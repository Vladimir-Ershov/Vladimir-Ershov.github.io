#!/usr/bin/env python3
"""
Script to prepare heartbeat sounds dataset from Kaggle.
Extracts and organizes audio files by their class prefix.
"""

import os
import zipfile
import shutil
from pathlib import Path
import argparse


def prepare_heartbeat_dataset(zip_path, output_dir="data/hb"):
    """
    Extract and organize heartbeat sounds dataset.

    Args:
        zip_path: Path to the downloaded Kaggle zip file
        output_dir: Directory where organized data will be stored
    """
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Temporary extraction directory
    temp_extract_dir = output_path / "temp_extract"
    temp_extract_dir.mkdir(exist_ok=True)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # Find all audio files (assuming .wav format, adjust if needed)
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.aiff', '.AIFF']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(temp_extract_dir.rglob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    # Organize files by class prefix
    class_counts = {}

    for audio_file in audio_files:
        # Get the filename without extension
        filename = audio_file.stem

        # Extract class prefix (everything before first underscore or digit)
        # Common patterns: artifact_01.wav, murmur_02.wav, normal_03.wav
        # Also handle patterns like: normal1.wav, murmur__201.wav
        if '_' in filename:
            class_name = filename.split('_')[0]
        elif '-' in filename:
            class_name = filename.split('-')[0]
        else:
            # Handle case where class name goes until first digit
            import re
            match = re.match(r'^([a-zA-Z]+)', filename)
            class_name = match.group(1) if match else filename

        # Clean up class name (lowercase, remove extra characters)
        class_name = class_name.lower().strip()

        # Create class directory
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)

        # Move file to class directory
        destination = class_dir / audio_file.name
        shutil.move(str(audio_file), str(destination))

        # Track counts
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Clean up temporary directory
    shutil.rmtree(temp_extract_dir)

    # Print summary
    print("\nDataset organization complete!")
    print(f"Output directory: {output_path.absolute()}")
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} files")

    return class_counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare heartbeat sounds dataset from Kaggle"
    )
    parser.add_argument(
        "zip_file",
        help="Path to the downloaded Kaggle dataset zip file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/hb",
        help="Output directory for organized dataset (default: data/hb)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.zip_file):
        print(f"Error: File '{args.zip_file}' not found")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds")
        print("Then run this script with the path to the downloaded zip file")
        return 1

    prepare_heartbeat_dataset(args.zip_file, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())