import torchaudio
import torch
import tempfile
import numpy as np

print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Available backends: {torchaudio.list_audio_backends()}")

# Test each backend with actual audio functionality
backends = torchaudio.list_audio_backends()

# Create a simple test audio file
sample_rate = 16000
duration = 1.0  # 1 second
t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave

print(f"\nGenerated test waveform: shape={waveform.shape}, dtype={waveform.dtype}")

for backend in backends:
    print(f"\n--- Testing {backend} backend ---")
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        # Test saving
        print(f"Saving audio with {backend} backend...")
        torchaudio.save(temp_path, waveform, sample_rate, backend=backend)
        print(f"✅ Save successful with {backend}")

        # Test loading
        print(f"Loading audio with {backend} backend...")
        loaded_waveform, loaded_sr = torchaudio.load(temp_path, backend=backend)
        print(f"✅ Load successful with {backend}")
        print(f"   Loaded: shape={loaded_waveform.shape}, sample_rate={loaded_sr}")

        # Clean up
        import os
        os.unlink(temp_path)

    except Exception as e:
        print(f"❌ Error with {backend} backend: {e}")

# Test the old 'sox_io' backend name specifically
print(f"\n--- Testing legacy 'sox_io' backend ---")
try:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name

    torchaudio.save(temp_path, waveform, sample_rate, backend='sox_io')
    loaded_waveform, loaded_sr = torchaudio.load(temp_path, backend='sox_io')
    print(f"✅ Legacy 'sox_io' backend works!")
    print(f"   Loaded: shape={loaded_waveform.shape}, sample_rate={loaded_sr}")

    import os
    os.unlink(temp_path)

except Exception as e:
    print(f"❌ Legacy 'sox_io' backend error: {e}")

print("\n=== Summary ===")
print(f"Available backends: {torchaudio.list_audio_backends()}")
print("All backends tested!")