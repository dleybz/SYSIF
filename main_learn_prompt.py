import argparse
import torch
import pandas as pd
from tqdm import tqdm
import logging
import random

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
    parser.add_argument('--n_iterations_max', type=int, default=1000)


    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":

    args = parse_args()
    
    random_seed = init_random_seed(args.seed)
    init_device(args.device)

    # load LM
    model_name = args.model_name
    model = CausalLanguageModel(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fast_tkn=True if not ('opt' in model_name) else False, #because of a bug in OPT
        fp16=args.fp16,
        padding_side='left')

    #load LAMA
    lamaset = LAMAset(args.lama_path, portion=0.05)

    #load human rephrases
    paraphrases=parse_paraphrases(args.paraphrase_path)        
    # only keep template where '[X]' is the first token / TODO: adress this
    paraphrases={relation:[t for t in templates if t.startswith('[X]')] for relation,templates in paraphrases.items()}

    # initialise the algo
    autoprompt = DiscreteGradientPromptSearch(model)

    for relation in lamaset.get_relations(): # in the future, run all relation in parallel in different scrips
        initial_template = random.sample(paraphrases[relation], 2)
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """
        autoprompt.train(initial_template, lamaset, relation, args.n_iterations_max, 4)
        # dev set