import torch
import os
import pickle
import logging
from src.fmap.fmap import extract_fmap
from src.data.dataset_loader import load_hf_dataset_with_sampling
from src.model.causal_lm import CausalLanguageModel



if __name__ == "__main__":

    name = "wikitext,wikitext-103-raw-v1,train"#'wikipedia,20220301.en,train'
    dataset = load_hf_dataset_with_sampling(name, n_samples=100)

    model_name = "EleutherAI/pythia-70m-deduped"  # You can replace this with the model name you want to use
    model = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    mode = ['input', 'output']

    unit_tokens_accum, tokens_count = extract_fmap(model=model, 
                 dataset=dataset,
                 batch_size=32,
                 window_size=15,
                 window_stride=15,
                 device='cpu',
                 mode=mode)
    
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
    exp_name = f'test_wikitext'
    exp_name += '_'+warning_flag
    save_dir = os.path.join('./',f'unit-token-wiki')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, f'tokens-count-{exp_name}.pickle'), 'wb') as handle:
        pickle.dump(tokens_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir,f'unit_tokens_accum-{exp_name}.pickle'), 'wb') as handle:
        pickle.dump(unit_tokens_accum, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done!')