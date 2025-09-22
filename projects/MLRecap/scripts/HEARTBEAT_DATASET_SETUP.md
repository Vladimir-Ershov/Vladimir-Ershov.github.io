# Heartbeat Sounds Dataset Setup

## Quick Start

The dataset has been successfully organized into `data/hb/` with the following structure:

```
data/hb/
├── artifact/     (40 files)
├── extrahls/     (19 files)
├── extrastole/   (46 files)
├── murmur/       (34 files)
├── normal/       (31 files)
└── unlabelled/   (10 files)
```

## To Download the Real Dataset

Since automatic download requires Kaggle API credentials, you need to:

1. **Manual Download (Easiest)**
   - Visit: https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds
   - Click the "Download" button (you may need to sign in)
   - Save the zip file to this directory

2. **Run the preparation script**
   ```bash
   python prepare_heartbeat_dataset.py <downloaded_file.zip>
   ```

## Alternative: Kaggle API Setup

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/`
4. Run:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d kinguistics/heartbeat-sounds
   python prepare_heartbeat_dataset.py heartbeat-sounds.zip
   ```

## Dataset Information

The script automatically:
- Extracts all audio files from the zip
- Identifies class from filename prefix (e.g., "normal_001.wav" → "normal" class)
- Creates separate folders for each class
- Handles various naming patterns (underscore, dash, or direct numbering)

## Current Status

A sample dataset with the same structure has been created and organized in `data/hb/`.
Replace it with the real Kaggle dataset when available.