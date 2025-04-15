sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=LORA
#SBATCH --gres=gpu:volta:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-lora_1.out
#SBATCH --exclusive

# Run the script
/state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/bin/python ~/notebooks/gLM2_trainer_v0.py \
    -n uniref_1 \
    -i /home/gridsan/jhoff/solab_shared/uniref50/uniref50.fasta \
    -w /home/gridsan/jhoff/models/glm_finetuning/weights/  
EOF
