import argparse
import torch
import pandas as pd
from tqdm import tqdm
import logging
import random
import os

from src.prompt.machine_prompt import DiscreteGradientPromptSearch
from src.prompt.utils import parse_paraphrases
from src.data.lama_dataset import LAMAset
from src.utils.init_utils import init_device, init_random_seed
from src.model.causal_lm import CausalLanguageModel
from src.data.dataset_loader import batchify


def parse_args():
    parser = argparse.ArgumentParser(description='AMAP')

    # Data selection
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda', help='Which computation device: cuda or mps')
    parser.add_argument('--output_dir', type=str, default='./amap', help='the output directory to store prediction results')
    parser.add_argument('--fp16', action='store_true', help='use half precision')
    parser.add_argument('--paraphrase_path', type=str, default='data/paraphrases/relation-paraphrases_v2.txt')
    parser.add_argument('--lama_path', type=str, default='data/opti-data/autoprompt_data')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--n_iterations_max', type=int, default=100)
    parser.add_argument('--n_population', type=int, default=50)
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--relation', type=str, default='all')

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":

    args = parse_args()
    
    random_seed = init_random_seed(args.seed)
    init_device(args.device)

    # load LM
    print("Loading model...")
    model_name = args.model_name
    model = CausalLanguageModel(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fast_tkn=True if not ('opt' in model_name) else False, #because of a bug in OPT
        fp16=args.fp16,
        padding_side='left')
    model_name_parse = model_name.split('/')[-1]

    #load LAMA
    print("Loading LAMA...")
    lamaset = LAMAset(args.lama_path, portion=1.0)

    #load human rephrases
    print("Loading human paraphrases...")
    paraphrases=parse_paraphrases(args.paraphrase_path)        
    # only keep template where '[X]' is the first token / TODO: adress this
    paraphrases={relation:[t for t in templates if t.startswith('[X]')] for relation,templates in paraphrases.items()}


    print("Starting!")
    # initialise the algo
    autoprompt = DiscreteGradientPromptSearch(model, args.n_population, args.num_candidates, n_rounds=3)

    relations = lamaset.get_relations() if args.relation=='all' else [args.relation,]
    # relations = ['P176',]

    for relation in relations: # in the future, run all relation in parallel in different scrips
        initial_template = paraphrases[relation]
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """
        savepath = os.path.join(args.output,f'disc-prompt-search_{model_name_parse}_{relation}_{random_seed}.tsv') 
        autoprompt.train(initial_template, lamaset, relation, args.n_iterations_max, args.batch_size, savepath)
        # dev set