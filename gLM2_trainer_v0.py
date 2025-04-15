from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM 
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
from math import floor
import time 
from prettytable import PrettyTable
import argparse

class GenomicDataset(Dataset):
    def __init__(self, fasta_path, tokenizer, max_length, masked_token_id, mlm_probability=0.30):
        self.sequences = self.load_fasta_entries(fasta_path)  # parse the FASTA
        self.tokenizer = tokenizer
        self.masked_token_id = masked_token_id
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
    def load_fasta_entries(self,fasta_path):
        """
        Reads a FASTA file and returns a list of sequences.

        Args:
            fasta_path (str): Path to the FASTA file.

        Returns:
            list: A list of sequence strings.
        """
        sequences = []
        with open(fasta_path, "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append(str(record.seq))  # Convert Seq object to string
        return sequences
    
    def mask_tokens(self, Input_ids, mask_token_id, mask_percent=.3):
        """
        Mask a percentage of tokens in the input for language modeling.

        This function selects tokens (that are not equal to 1) at random, replaces
        them with `mask_token_id` in `masked_input_ids`, and sets the corresponding
        positions in `labels` to their original token IDs. All unmasked tokens in
        `labels` are set to `-100`, indicating they should be ignored during the
        language modeling loss computation.

        Args:
            input_ids (torch.Tensor):
                The original input tensor of shape (batch_size, seq_length) or
                (1, seq_length). Tokens with ID=1 are assumed to be "non-maskable"
                (e.g., special tokens).
            mask_token_id (int):
                The token ID to use for masking.
            mask_percent (float, optional):
                Fraction of maskable tokens to replace with `mask_token_id`.
                Defaults to 0.3.

        Returns:
            masked_input_ids (torch.Tensor):
                A copy of `input_ids` with `mask_percent` fraction of eligible
                tokens replaced by `mask_token_id`.
            labels (torch.Tensor):
                A copy of the original `input_ids`, except that all unmasked tokens
                are set to -100, making them not contribute to the language model
                loss.
        """
        labels = Input_ids.clone() 
        input_ids = Input_ids.clone()
        masks = torch.zeros(Input_ids.shape[1])
        
        maskable_idxs = torch.where(labels!=1)[-1]
        maskable_len = maskable_idxs.shape[-1]
        mask_num = round(maskable_len * .3)
        mask_idxs = torch.tensor(np.random.choice(maskable_len,mask_num,replace=False))
        input_ids.flatten()[mask_idxs] = mask_token_id
        
        unmasked_idxs = [i for i in list(range(0,labels.shape[-1])) if i not in mask_idxs.numpy()]
        labels.flatten()[unmasked_idxs] = -100
        
        masks[mask_idxs] = 1
        
        return input_ids, labels, masks


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        encoding = self.tokenizer(seq_str, max_length=self.max_length,
                                  truncation=True, padding="max_length",
                                  return_tensors="pt")
        input_ids, labels, masks = self.mask_tokens(encoding["input_ids"], 
                                        self.masked_token_id, 
                                        self.mlm_probability)

        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),  
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "masks": masks.squeeze(0)
        }    
# ------------------------- Initialize ---------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Training Loop for gLM2 LORA finetuning.")
    parser.add_argument('-n','--name', type=str, required=True, help='Name of training run')
    parser.add_argument('-i','--fasta', type=str, required=True, help='path/to/dataset.fasta')
    parser.add_argument('-w','--weights_path', type=str, required=True, help='LORA weights path.')
    parser.add_argument('-o','--log_path', type=str, default='/home/gridsan/jhoff/models/glm_finetuning/training', help='WandB log path [Default: w]')
    parser.add_argument('-x','--offline', type=bool, default=True, help='WandB offline run [Default: False]')
    parser.add_argument('-b','--batch_size', type=int, default=16, help='Batch Size [Default: 16]')
    parser.add_argument('-r','--rank', type=int, default=4, help='LORA rank size [Default: 4]')
    parser.add_argument('-d','--dropout', type=float, default=0.15, help='LORA rank size [Default: 0.15]')
    parser.add_argument('-l','--lr', type=float, default=0.001, help='Learning Rate [Default: 1e-3]')
    parser.add_argument('-e','--epoch', type=int, default=2, help='Number of Epochs [Default: 2]')
    parser.add_argument('-t','--train', type=int, default=0.7, help='Training Ratio [Default 0.7]. Val/test = (1 - Train_ratio)/2')
    
    args = parser.parse_args()
    return args

def load_model(Rank, Dropout):
    global DEVICE, model, model_pft, tokenizer, MASK_TOKEN_ID
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'>> DEVICE: {DEVICE}')
    glm_path = '/data1/groups/solab/glm_models/gLM2_650M'
    print(f'>> Loading gLM2: {glm_path}')
    model = AutoModelForMaskedLM.from_pretrained(glm_path,trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE) #AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(glm_path,trust_remote_code=True)
    MASK_TOKEN_ID = tokenizer.mask_token_id
    print(f'>> Loading LORA: r:{Rank}, lora_dropout:{Dropout}')
    config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["wqkv"],
    r=Rank,
    lora_dropout=Dropout)
    model_pft = get_peft_model(model, config) #dont want the outer wraper, use base model (has LORA but not the wrapper)
    print(model_pft)

# ------------------------- Load Dataset -------------------------------

def dataset_loader(Path, Max_length=128):
    Dataset = GenomicDataset(
    fasta_path=Path,
    tokenizer=tokenizer,
    masked_token_id=MASK_TOKEN_ID,
    max_length=Max_length)
    return Dataset

def display_data_dimension_table(train_loader, val_loader, test_loader):
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    table = PrettyTable()
    table.field_names = ["Loader", "Dataset Size", "Batch Size", "Batches per Epoch"]
    
    table.add_row(["Train Loader", train_size, 32, len(train_loader)])
    table.add_row(["Validation Loader", val_size, 32, len(val_loader)])
    table.add_row(["Test Loader", test_size, 32, len(test_loader)])

    print(table)

def split_dataset(Dataset_Path, train_ratio, Batch_size):
    dataset = dataset_loader(Dataset_Path)
    
    val_ratio = (1-train_ratio)/2
    test_ratio = val_ratio
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure total size matches

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, pin_memory=True, prefetch_factor=2, num_workers=16) #pin_memory=True, prefetch_factor=2, num_workers=16 #load ahead
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)
    
    display_data_dimension_table(train_loader, val_loader, test_loader)
    return train_loader, val_loader, test_loader 

def config_wandb(name, learning_rate, dataset, num_epochs, batch_size, log_path, lora_rank, dropout, train_ratio, status):
    if status == True: status = 'offline'
    else: status = True
    print(f'>> wandb status: {status}')
    #wandb.finish() #otherwise may not write
    run = wandb.init(
        name=name,
        project="glm2_LORA",
        mode=status, 
        dir=log_path,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": log_path,
            "dataset": dataset,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lora_rank":lora_rank,
            "dropout":dropout,
            "train_ratio":train_ratio
        },
        settings=wandb.Settings(init_timeout=360)
    )
    if status == 'offline': print(f'>> Running Wandb offline, consider running "python ~/notebooks/wandb_helper.py -o {log_path}/wandb/latest-run"')
    return run

# ----------------------- Training Loop ------------------------

def compute_loss(logits, labels, mask):
    """Compute masked loss for MLM."""
    loss_fn = nn.CrossEntropyLoss(reduction='mean')  
    loss = loss_fn(logits.transpose(1, 2), labels)  
    return loss

def train_one_epoch(Model, dataloader, optimizer, Run, Lora_save_path):
    Model.train()
    total_loss = 0

    for i, batch in tqdm(enumerate(dataloader),total=len(dataloader),desc='Training batch'):

        # --- Data to device
        inputs_batch = batch['input_ids'].to(DEVICE)
        attention_masks_batch = batch['attention_mask'].to(DEVICE)
        true_labels_batch = batch['labels'].to(DEVICE)
        mask_batch = batch['masks'].to(DEVICE)

        # --- Forward pass
        outputs = Model(input_ids=inputs_batch, attention_mask=attention_masks_batch)
        logits = outputs.logits.squeeze(-1)

        # --- Compute loss
        mean_loss = compute_loss(logits, true_labels_batch, mask_batch)

        # --- Backprop & optimize
        mean_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)

        total_loss += mean_loss.item()
        
        if i % 10 == 0:
            avg_loss = total_loss / (i + 1)
            Run.log({"avg_train_loss": avg_loss, "mean_train_loss":mean_loss})
            
        if i % 1000: #and i != 0:
            model.save_pretrained(Lora_save_path) # Save the og model, not the peft? It seems like peft isnt being saved?
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(Model, dataloader):
    """Validate the model."""
    Model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader),total=len(train_loader),desc='Validation batch'):
            inputs_batch, attention_masks_batch, true_labels_batch, mask_batch = batch['input_ids'].to(DEVICE) , batch['attention_mask'].to(DEVICE) , batch['labels'].to(DEVICE) , batch['masks'].to(DEVICE)

            # Forward Model
            outputs = model(input_ids=inputs_batch, attention_mask=attention_masks_batch)
            logits = outputs.logits.squeeze(-1)

            # Compute loss
            mean_loss = compute_loss(logits, true_labels_batch, mask_batch)

            total_loss += mean_loss.item()
            
            if i % 10 == 0:
                avg_loss = total_loss / (i + 1)
                Run.log({"avg_val_loss": avg_loss, "mean_val_loss":mean_loss})

        average_loss = total_loss / len(dataloader)
    return average_loss

def train_model(Model, train_dataloader, test_dataloader, num_epochs, optimizer, lora_save_path, Run):
    """Train and validate the model."""
    T0 = time.time()
    os.makedirs(lora_save_path, exist_ok=True)
    for epoch in range(num_epochs):
        print(f">> Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(Model, train_dataloader, optimizer, Run, lora_save_path)
        test_loss = validate(Model, test_dataloader)
        Print(f'>>> Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f})')
        
        Model.save_pretrained(lora_save_path)
        print(f'>>> Updated LORA weights saved to: {lora_save_path}')
   
        # Log metrics to wandb.
        Run.log({"epoch_train_loss": train_loss, "epoch_val_loss": test_loss})
    
    # Finish the run and upload any remaining data.
    Run.finish()
    
    print(f">> Training Loop Complete! Elapsed Time: {((time.time()-T0)/60)} min")

# ------------------------- Main ---------------------------------

def main():
    args = get_args() #change to config file
    print('> Loading Model:')
    load_model(args.rank, args.dropout)
    print('> Loading Dataset:')
    train_loader, val_loader, test_loader = split_dataset(args.fasta, args.train, args.batch_size)
    print('> Configuring wandb:')
    run = config_wandb(args.name,args.lr,args.fasta,args.epoch, args.batch_size, args.log_path, args.rank, args.dropout, args.train, args.offline)
    print('> Starting Training Loop:')
    optimizer = optim.Adam(model_pft.base_model.parameters(), lr=args.lr)
    train_model(model, train_loader, val_loader, args.epoch, optimizer, args.weights_path, run) # model.base_model if peft
    #test?
    print('> Finetuning Workflow Complete.')

if __name__ == '__main__':
    main()
