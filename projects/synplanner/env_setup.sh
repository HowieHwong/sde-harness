#!/usr/bin/env bash
set -eo pipefail

conda create -y -n synplanner python=3.10

conda activate synplanner

conda install -y -c pytorch -c nvidia pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1

conda install -y -c conda-forge "numpy>=1.26.4" "typing-extensions>=4.12.2" rdkit=2023.9.6

pip install rdchiral==1.1.0 syntheseus==0.5.0 openai==1.99.9
