# glm_finetuning

Run ```python wandb_helper.py -o /home/gridsan/jhoff/models/glm_finetuning/training/wandb/latest-run``` in sbatch if running offline (slurm) to update WanB.

Update .sh for changing workflow.
```
Training Loop for gLM2 LORA finetuning.

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Name of training run
  -i FASTA, --fasta FASTA
                        path/to/dataset.fasta
  -w WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        LORA weights path.
  -o LOG_PATH, --log_path LOG_PATH
                        WandB log path [Default: w]
  -x OFFLINE, --offline OFFLINE
                        WandB offline run [Default: False]
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size [Default: 16]
  -r RANK, --rank RANK  LORA rank size [Default: 4]
  -d DROPOUT, --dropout DROPOUT
                        LORA rank size [Default: 0.15]
  -l LR, --lr LR        Learning Rate [Default: 1e-3]
  -e EPOCH, --epoch EPOCH
                        Number of Epochs [Default: 2]
  -t TRAIN, --train TRAIN
                        Training Ratio [Default 0.7]. Val/test = (1 - Train_ratio)/2
```
