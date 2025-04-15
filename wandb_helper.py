import argparse
import time
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Helper Script for Weights & Biases for training on jupyter (offline).")
    parser.add_argument('-o','--outpath', type=str, default='/home/gridsan/jhoff/models/glm_finetuning/training/wandb/latest-run/'
                        , required=False,  help='path/to/wandb/latest-run/')
    parser.add_argument('-t','--sleep', type=int, default=60, help='Seconds until refresh weights and balance [Default 60].')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    t0, k = time.time(), 0
    while True:  
        print(f'> Update Number: {k}; Time Elapsed: {((time.time()-t0)/60):.2f} min')
        subprocess.run(["wandb", "sync", args.outpath])
        time.sleep(args.sleep) 
        k+=1

if __name__ == '__main__':
    main()