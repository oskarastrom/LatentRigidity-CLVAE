import json
import pandas as pd
import os

# Custom
from config import parse_args, process_args
from dataloader import get_dataloaders
from models import cvaecaposr
from utils import get_logger, get_callbacks, generate_save_name
from main import main

dataset_variance_settings = {
    'mnist': 0.1,
    'cifar10': 1.0,
    'svhn': 0.01,
    'tiny_imagenet': 0.01,
    'cifar+10': 0.01,
    'cifar+50': 0.1
}

def main_batch(args):
    if args.mode == "train":
        dataset_list = ['mnist', 'cifar10', 'svhn', 'tiny_imagenet', 'cifar+10', 'cifar+50']
        if args.dataset != "": dataset_list = [args.dataset]
        for dataset in dataset_list:
            base_path = "checkpoints/"
            args.dataset = dataset
            args.t_var_scale = dataset_variance_settings[dataset]

            os.makedirs(base_path + dataset, exist_ok=True)
            splits = [0,1,2,3,4]
            for split_num in splits:
                args.split_num = split_num
                args.known_classes = ""
                args.unknown_classes = ""
                args = process_args(args)
                print(args.dataset, args.known_classes, args.unknown_classes)

                os.makedirs(base_path + dataset + "/" + str(args.split_num), exist_ok=True)

                experiments = ['fa', 'lc', 'lcf8', 'lcf64'] # run with learned variance: 'fa', 'fav', 'lc', 'lcv', 'lcf8', 'lcvf8', 'lcf64', 'lcvf64'
                for run in experiments:
                    args = update_args(args, run)
                    args.save_name = dataset + "_" + run
                    print(run, args.split_num)
                    main(args)

    elif args.mode == "test":
        base_path = "checkpoints/"
        dataset = args.dataset
        dataset_path = base_path + dataset + "/" + str(args.split_num) + "/"
        runs = ['fa', 'lc', 'lcv', 'lcf8', 'lcvf8', 'lcf64', 'lcvf64']

        results = []
        for run in runs:
            args = update_args(args, run)
            args.checkpoint = dataset_path + dataset + "_" + run + "_best.ckpt"
            print(args)
            testout_best = main(args)
            args.checkpoint = dataset_path + dataset + "_" + run + "_last.ckpt"
            testout_last = main(args)
            results.append({'name': run, 'best': testout_best[0]['test_auroc'], 'last': testout_last[0]['test_auroc']})
        df = pd.DataFrame(results)
        print(df)


def update_args(args, run_name):
    if run_name.startswith('fa'):
        args.learned_variance = "v" in run_name
        args.fixed_dim = 128
    elif run_name.startswith('lcv'):
        args.learned_variance = True
        if run_name.startswith('lcvf'):
            args.fixed_dim = int(run_name[4:])
        else:
            args.fixed_dim = 0
    elif run_name.startswith('lc'):
        args.learned_variance = False
        if run_name.startswith('lcf'):
            args.fixed_dim = int(run_name[3:])
        else:
            args.fixed_dim = 0
    return args

if __name__ == '__main__':

    # Parse args
    args = parse_args()

    # Main
    main_batch(args)




