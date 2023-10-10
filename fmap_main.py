import torch
import os
import pickle
import logging
import argparse
import random

from src.fmap.fmap import extract_fmap
from src.data.dataset_loader import load_hf_dataset_with_sampling
from src.model.causal_lm import CausalLanguageModel
from src.utils.init_utils import init_device, init_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description='OPTCorpus generation')

    # Data selection
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument('--dataset', type=str, default='wikipedia,20220301.en,train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--window_stride', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda', help='Which computation device: cuda or mps')
    parser.add_argument('--output_dir', type=str, default='./unit-token-analyze', help='the output directory to store prediction results')
    parser.add_argument('--fp16', action='store_true', help='use half precision')
    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":

    args = parse_args()
    
    random_seed = init_random_seed(args.seed)
    init_device(args.device)

    dataset = load_hf_dataset_with_sampling(args.dataset, n_samples=args.n_samples)

    model_name = args.model_name
    model = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu", fp16=args.fp16)

    mode = ['input', 'output']

    unit_tokens_accum, tokens_count = extract_fmap(model=model, 
                 dataset=dataset,
                 batch_size=args.batch_size,
                 window_size=args.window_size,
                 window_stride=args.window_stride,
                 device=args.device,
                 mode=mode, 
                 fp16=args.fp16)
    unit_tokens_accum = unit_tokens_accum.cpu()
    tokens_count = tokens_count.cpu()
    
    # safety check
    warning_flag = ''
    if 'input' in mode and 'output' in mode:
        try:
            input_token_sum = tokens_count['input'].sum(0)
            output_token_sum = tokens_count['output'].sum(0)
            assert input_token_sum == output_token_sum
        except AssertionError:
            logging.warning(f'Input and output do not have the same number of tokens! Input:{input_token_sum}. Output:{output_token_sum}')
            warning_flag += 'wTkn'
        try:
            total_input = (unit_tokens_accum['output'][0] * tokens_count['output'].unsqueeze(-1)).sum()
            total_output = (unit_tokens_accum['output'][0] * tokens_count['output'].unsqueeze(-1)).sum()
            assert abs(total_input - total_output) < 1
        except AssertionError:
            logging.warning(f'Input and output do not have the same total activation! Input:{total_input}. Output:{total_output}')
            warning_flag += 'wAct'
    
    # Save with pickle
    print('Saving stats...')
    exp_name = f'{args.model_name.split("/")[-1]}-N{args.n_samples}-{random_seed}'
    exp_name += '_'+warning_flag
    save_dir = os.path.join(args.output_dir,f'unit-token-wiki.{random_seed}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, f'tokens-count-{exp_name}.pickle'), 'wb') as handle:
        pickle.dump(tokens_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir,f'unit_tokens_accum-{exp_name}.pickle'), 'wb') as handle:
        pickle.dump(unit_tokens_accum, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done!')