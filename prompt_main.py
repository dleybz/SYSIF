import argparse
import torch
import pandas as pd
from tqdm import tqdm
import logging

# from src.prompt.machine_prompt import EvoMachinePrompt, init_char_crossover, init_char_mutate, init_lama_fitness
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
    parser.add_argument('--n_generated_tokens', type=int, default=2)


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
    lamaset = LAMAset(args.lama_path)

    #load human rephrases
    paraphrases=parse_paraphrases(args.paraphrase_path)           

    # store evaluation results on a dataframe
    df_evaluation = pd.DataFrame()

    # construct the prompts
    print('Constructing the prompts...')
    df_prompts = []
    for relation in lamaset.get_relations():
        lama_label = lamaset.info[lamaset.info['relation']==relation]['label'].item()
        try:
            for tid, this_template in enumerate(paraphrases[relation]):
                # this_template = f'Source: Wikipedia. Label: {lama_label}. '+this_template
                # this_template = f'Label: {lama_label}. '+this_template
                # this_template = f'Source: Wikipedia. '+this_template
                # fill the template with LAMA's objects
                filled_list = lamaset.fill_template(relation, this_template, set='dev')
                df_temp = pd.DataFrame()
                df_temp['prompt'] = [tp[0] for tp in filled_list]
                df_temp['label'] = [tp[1] for tp in filled_list]
                df_temp['relation'] = [relation,] * len(df_temp)
                df_temp['template_id'] = [f'{relation}_{tid}',] * len(df_temp)
                df_prompts.append(df_temp)
        except KeyError:
            logging.warning(f'[EVALUATION] Paraphrase does not contains the relation {relation}. Skipping it.')
    df_prompts = pd.concat(df_prompts)

    # while(True):
    #     df_sample = df_prompts.sample(10)
    #     print("========== Samples:")
    #     print(df_sample)
    #     print("========== Samples:")
    #     input('')


    # feed prompts to the LM and gather predictions
    prompt_list = df_prompts['prompt'].values.tolist()
    pred_list = []
    batches, n_batches = batchify(prompt_list, args.batch_size, drop_last=False, output_text=True)
    for batch in tqdm(batches, desc="[EVALUATION]"):
        text_generated = model.generate_tokens_from_text_batch(batch, n_tokens=args.n_generated_tokens)
        pred_list += text_generated
    df_prompts['pred'] = pred_list

    df_prompts.to_csv(model.model_name.replace('/', '_')+'_paraphrase_eval.csv')

    # evaluate paraphrases + the model

    df_prompts['correct'] = df_prompts.apply(lambda row: row['label'] in row['pred'], axis=1)    
    macro_all = df_prompts['correct'].mean()
    print(macro_all)
    
    # take the best template for each relation, then average
    micro_max = df_prompts.groupby(['relation', 'template_id'])['correct'].mean().groupby('relation').max().mean()
    print(micro_max)
    micro_mean = df_prompts.groupby('relation')['correct'].mean().mean()
    micro_min = df_prompts.groupby('relation')['correct'].min().mean()

    # get best templates
    idmax = df_prompts.groupby(['relation', 'template_id'])['correct'].mean().groupby(['relation']).idxmax()
    best_templates = [rel[-1] for rel in idmax]
    df_best = df_prompts[df_prompts['template_id'].isin(best_templates)]
    df_best.groupby(['relation', 'template_id'])['correct'].mean().mean()

    # averaged per label accuracy
    micro_max_balanced = df_prompts.groupby(['relation', 'label'])['correct'].mean().mean()

        # accuracy top1, top5, f1score
    