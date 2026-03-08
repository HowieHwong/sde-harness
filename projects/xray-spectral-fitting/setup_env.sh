#!/bin/bash
# Setup script for X-Ray Spectral Fitting Benchmark
#
# This script creates a conda environment with Sherpa and XSPEC support.
#
# Usage:
#   ./setup_env.sh
#
# After running, activate with:
#   conda activate xray-spectral-fitting

set -e

ENV_NAME="xray-spectral-fitting"

echo "=============================================="
echo "X-Ray Spectral Fitting Environment Setup"
echo "=============================================="

# Detect architecture
ARCH=$(uname -m)
OS=$(uname -s)

echo "Detected: $OS on $ARCH"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Exiting. Activate existing environment with: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create environment based on platform
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo ""
    echo "Apple Silicon detected. Creating x86_64 environment via Rosetta..."
    echo "(This may take a few minutes)"
    echo ""
    
    # Create x86_64 environment for Apple Silicon
    CONDA_SUBDIR=osx-64 conda create -n $ENV_NAME python=3.10 -y
    
    # Configure environment to stay x86_64
    conda activate $ENV_NAME
    conda config --env --set subdir osx-64
    
    # Install Sherpa with XSPEC
    conda install -c conda-forge -c ciao sherpa -y
else
    echo ""
    echo "Creating native environment..."
    echo ""
    
    # Create native environment
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
    
    # Install Sherpa with XSPEC
    conda install -c conda-forge -c ciao sherpa -y
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install openai

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use the environment:"
echo "  1. Activate:  conda activate $ENV_NAME"
echo "  2. Set API key: export OPENAI_API_KEY='your-key'"
echo "  3. Run: python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha -v"
echo ""
echo "To verify installation:"
echo "  python -c \"from sherpa.astro import ui; ui.set_xsxsect('vern'); print('OK')\""
echo ""
