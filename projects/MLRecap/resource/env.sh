#!/usr/bin/env bash
set -euo pipefail

# 1. Install Pixi (Linux/macOS)
echo "Installing Pixi..."
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Optional: disable Conda auto-activate for base
echo "Disabling Conda base auto-activation..."
conda config --set auto_activate_base false

# 3. Source shell config to apply PATH changes
echo "Sourcing shell configuration..."
if [ -n "${ZSH_VERSION-}" ]; then
  source ~/.zshrc
elif [ -n "${BASH_VERSION-}" ]; then
  source ~/.bashrc
else
  echo "Shell not recognized. Please source your shell config manually."
fi

echo "Pixi installation complete. Pixi is now your default package manager."
