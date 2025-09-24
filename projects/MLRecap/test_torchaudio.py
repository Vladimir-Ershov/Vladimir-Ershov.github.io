import torchaudio
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Installation path: {torchaudio.__file__}")
print(f"Available backends: {torchaudio.list_audio_backends()}")

# Check if SOX and other backends are available
try:
    print("Testing SOX backend...")
    import torchaudio.backend.sox_backend
    print("SOX backend module exists")
except ImportError as e:
    print(f"SOX backend import error: {e}")

try:
    print("Testing soundfile backend...")
    import torchaudio.backend.soundfile_backend
    print("soundfile backend module exists")
except ImportError as e:
    print(f"soundfile backend import error: {e}")