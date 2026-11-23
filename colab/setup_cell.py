"""
Colab setup cell — run this first in every notebook.
Pins library versions to avoid transformers 5.x breaking anticipation.
Does NOT reinstall torch (uses Colab's preinstalled CUDA build).
"""
import subprocess, sys

def run(cmd: str) -> None:
    result = subprocess.run(cmd, shell=True, check=True,
                            capture_output=True, text=True)
    print(result.stdout[-2000:] if result.stdout else "")

# 1. System packages for WAV rendering (fluidsynth + GM soundfont)
run("apt-get install -qq -y fluidsynth fluid-soundfont-gm")
run("pip install -q pyfluidsynth")

# 2. Pin transformers stack
run("pip install -q transformers==4.57.6 huggingface_hub==0.36.0 "
    "accelerate==1.12.0 'peft==0.18.*'")

# 3. Install anticipation without touching torch
run("pip install -q --no-deps "
    "git+https://github.com/jthickstun/anticipation.git")

# 4. MIDI / audio / ML utilities
run("pip install -q mido==1.2.10 pretty_midi scipy scikit-learn pyyaml")

print("Setup complete.")
