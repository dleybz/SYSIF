import torch
import os
import pickle
import logging
import argparse
import random

from src.amap.amap import LMamap
from src.data.dataset_loader import load_hf_dataset_with_sampling
from src.model.causal_lm import CausalLanguageModel
from src.utils.init_utils import init_device, init_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description='AMAP')

    # Data selection
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument('--dataset', type=str, default='wikipedia,20220301.en,train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--window_stride', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda', help='Which computation device: cuda or mps')
    parser.add_argument('--output_dir', type=str, default='./amap', help='the output directory to store prediction results')
    parser.add_argument('--pos', action='store_true', help='Include token position in the amap')
    parser.add_argument('--fp16', action='store_true', help='use half precision')
    parser.add_argument('--extract', action='store_true', help='Run amap extraction')
    parser.add_argument('--load', type=str, default='', help='Load the amap in the provided folder')


    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":

    args = parse_args()
    
    random_seed = init_random_seed(args.seed)
    init_device(args.device)

    model_name = args.model_name
    model = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu", fp16=args.fp16)

    mode = ['input', 'output']

    amapper = LMamap(model=model,
                     device=args.device,
                     mode=mode,
                     fp16=args.fp16)

    if args.load != '':
        print('[AMAP] Loading an existing amap...')
        amapper.load(args.load, args.dataset)
        """
        amapper.amap: the actication maps (a dictionary with different modes, e.g. ['input', 'output'])
                      each map is a list of n_layers tensors with shape (vocab_size, n_units)
        amapper.tokens_count: tokens count (also a dictionary)
        amapper.vocab_list: the list of tokens (the index is aligned with amaps)
        
        if positions are tracked (check flags in amapper.special_tracking), the position id correspond to:
        position i = amapper.position_offset + i
        """
        

    elif args.extract:

        print('[AMAP] Starting extraction.')

        if args.pos:
            amapper.add_position(args.window_size)

        dataset = load_hf_dataset_with_sampling(args.dataset, n_samples=args.n_samples)

        amap, tokens_count, n_sentences = amapper.extract(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    window_size=args.window_size,
                    window_stride=args.window_stride)
        
        # safety check
        warning_flag = amapper.sanity_check(n_sentences)

        # Save with pickle
        print('Saving stats...')
        exp_name = f'{args.model_name.split("/")[-1]}-{args.dataset}-N{args.n_samples}-{random_seed}'
        if args.pos:
            exp_name += '_position'
        exp_name += '_'+warning_flag
        save_dir = os.path.join(args.output_dir,f'amap.{random_seed}')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, f'tokens-count-{exp_name}.pickle'), 'wb') as handle:
            pickle.dump(tokens_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_dir,f'amap-{exp_name}.pickle'), 'wb') as handle:
            pickle.dump(amap, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Done!')