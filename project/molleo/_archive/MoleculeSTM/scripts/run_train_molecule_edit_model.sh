#!/bin/bash
#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=me
#SBATCH --qos=m2
#SBATCH --partition=a40
#SBATCH --output=./slurm_files/%j.out
#SBATCH --error=./slurm_files/%j.error

export PATH=/pkgs/anaconda3/bin:$PATH
source $HOME/.zshrc
conda init bash
conda activate MoleculeSTM

python downstream_02_molecule_edit_step_01_MoleculeSTM_Space_Alignment.py \
  --MoleculeSTM_model_dir "/scratch/ssd004/scratch/mskrt/huggingface_models/MoleculeSTM/demo/demo_checkpoints_SMILES" \
  --output_model_dir "/scratch/ssd004/scratch/mskrt/huggingface_models/MoleculeSTM/demo/demo_checkpoints_SMILES/molecule_editing_100e"
