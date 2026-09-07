#!/bin/bash
# Convenience wrapper around `conda env create -f environment.yml` for the
# X-Ray Spectral Fitting benchmark. environment.yml is the single source of
# truth for the environment; this script only adds an "already exists?" check.
#
# Usage:
#   ./setup_env.sh
#   conda activate ScienceBench_XraySpectralFitting

set -euo pipefail

ENV_NAME="ScienceBench_XraySpectralFitting"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Install Miniforge/Miniconda first: https://github.com/conda-forge/miniforge"
    exit 1
fi

case "$(uname -s)/$(uname -m)" in
    Linux/x86_64|Darwin/x86_64|Darwin/arm64) ;;
    *)
        echo "Warning: Sherpa with XSPEC models is only published for linux-64, osx-64 and osx-arm64."
        echo "         Detected $(uname -s)/$(uname -m); the environment solve will most likely fail."
        ;;
esac

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Remove it first to recreate:"
    echo "  conda env remove -n $ENV_NAME"
    exit 0
fi

conda env create -f "$HERE/environment.yml"

echo
echo "Done. Next steps:"
echo "  conda activate $ENV_NAME"
echo "  python -c \"from sherpa.astro import ui; ui.set_xsxsect('vern'); print('Sherpa+XSPEC OK')\""
echo "  (then follow README.md, Install step 4, to create models.yaml / credentials.yaml)"
