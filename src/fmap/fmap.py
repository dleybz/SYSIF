import sys
sys.path.append('../')
import torch
import logging
from src.model.causal_lm import CausalLanguageModel
from src.data.dataset_loader import tokenize_slice_batch, load_hf_dataset_with_sampling

from tqdm import tqdm

def update_token_unit(unit_tokens, kn_act, layer, unique_id, token_ids, tokens_count, old_token_count, device):
    '''
    unit_tokens: Table [n_layer, n_tokens, n_kn] where the accummulated value of token/activation pairs is stored. 
    kn_act: knowledge neuron activations
    layer: id of the layer
    unique_id: set of the tokens id in the input (resp. output) sequence (only unique tokens)
    tokens_id: list of the tokens id the input (resp. output) sequence (one token can appear many times)
    tokens_count: count the number of time each token appeared
    old_token_count: previous tokens_count, used to compute the cumulative average
    device: which device to use
    In order to reduce computation, only updates the tokens in unique_id (i.e. those in the input sequence)
    '''
    save_device = unit_tokens[layer].device
    kn_d1, kn_d2, kn_d3 = kn_act[layer].shape
    # create an index mask, to only process tokens in the batch. [n_tokens_in_the_input, n_unique_tokens_in_the_input]
    expand_unique_id = unique_id.unsqueeze(0).expand(token_ids.size(0), -1)
    # in the mask, line i correspond to the location of unique token i in the input (resp. output) sequence
    index_mask = (expand_unique_id == token_ids.unsqueeze(-1)).t().type(DTYPE) 
    # compute the unit-token activations for the batch, on the device
    """
    index_mask: [n_uniques_in_sequence, n_tokens_in_sequence]. Line i has 1 where token i appear in the sequence
    kn_act[layer].view(kn_d1*kn_d2, kn_d3): [n_tokens_in_sequence, n_feats]

    batch_unit_token_cum: index_mask x kn_act[layer] = [n_uniques_in_sequence, n_feats].
    Line i correspond to the accumulated kn features obtained when token i is the input.
    """
    batch_unit_token_cum = torch.matmul(index_mask.to(device), kn_act[layer].to(device).view(kn_d1*kn_d2, kn_d3))
    # update the cumuluative average
    unit_tokens[layer][unique_id] = cumulative_average(
        new_item    = batch_unit_token_cum,
        new_count   = tokens_count[unique_id].unsqueeze(-1),
        old_count   = old_token_count[unique_id].unsqueeze(-1),
        old_average = unit_tokens[layer][unique_id.to(save_device)],
        device      = device,
    ).to(save_device)

    return unit_tokens

def cumulative_average(new_item, new_count, old_count, old_average, device='cpu'):
    new_item = new_item.to(device)
    new_count = new_count.to(device)
    old_count = old_count.to(device)
    old_average = old_average.to(device)
    return (new_item + (old_count) * old_average) / (new_count)

def extract_fmap(model: CausalLanguageModel, dataset, batch_size, window_size, window_stride, device, mode=['input', 'output'], fp16=True):
    DTYPE=torch.float16 if fp16 else torch.float32
    # dataset: hugging face dataset
    # model: hugging face model
    kn_act_buffer = model.enable_output_knowledge_neurons()
    vocab_list = model.get_vocab()
    n_layers = model.get_nb_layers()
    n_units = model.get_nb_knowledge_neurons(layer_id=0)
    # preprocess dataset
    dataset_sliced_batched, n_batch = tokenize_slice_batch(dataset, model.tokenizer, batch_size, window_size, window_stride)
    # init token/unit matrix
    '''
    unit_tokens_accum['input'][l][i][j]
    provides the accumulated activation of knowledge neuron j when vocab item i is the last token of the input
    '''
    unit_tokens_accum = {}
    if 'input' in mode:
        unit_tokens_accum['input'] = [torch.full(size=(len(vocab_list), n_units), fill_value=0.0, dtype=DTYPE) for l in range(n_layers)]
    if 'output' in mode:
        unit_tokens_accum['output'] = [torch.full(size=(len(vocab_list), n_units), fill_value=0.0, dtype=DTYPE) for l in range(n_layers)]
    tokens_count = {'input':torch.zeros(size=(len(vocab_list),)).int().to(device), 'output':torch.zeros(size=(len(vocab_list),)).int().to(device)}
    
    # iterate on the dataset
    for input_ids, attention_mask in tqdm(dataset_sliced_batched, total=n_batch):
        d_batch, d_sent = input_ids.shape
        
        # forward pass
        output = model.forward_pass_nograd((input_ids, attention_mask), tokenize=False)
        
        # accumulate input and output token ids
        tokens_ids = {
            'input': input_ids.flatten().to(device).type(DTYPE),
            'output': torch.argmax(output.logits.detach(), dim=-1).flatten().to(device).type(DTYPE)
        }
        # count unique tokens
        unique_id = {}
        old_token_count = {}

        for mode in unit_tokens_accum.keys():
            """
            Detect the unique tokens_id in the input sequence, and count them.
            Update the total tokens_count.
            Save an old_tokens_count, used to compute the cumulative average
            """
            unique_tokens_with_count = torch.unique(tokens_ids[mode], return_counts=True)
            unique_id[mode] = unique_tokens_with_count[0].long() # set of unique tokens in the input (resp. output)
            count_id = unique_tokens_with_count[1] # corresponding count for each unique token
            old_token_count[mode] = tokens_count[mode].clone().detach() # save old count
            tokens_count[mode][unique_id[mode]] += count_id # update new count

        # for each layer accumulate the unit-token association
        for l in range(n_layers):
            # per token stats
            for mode in unit_tokens_accum.keys():
                with torch.no_grad():
                    unit_tokens_accum[mode] = update_token_unit(
                        unit_tokens=unit_tokens_accum[mode],
                        kn_act=kn_act_buffer.type(DTYPE),
                        layer=l,
                        unique_id=unique_id[mode],
                        token_ids=tokens_ids[mode],
                        tokens_count=tokens_count[mode],
                        old_token_count=old_token_count[mode],
                        device=device,)
                
    return unit_tokens_accum, tokens_count       
