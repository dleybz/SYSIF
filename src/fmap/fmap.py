import sys
sys.path.append('../')
import torch
import logging
from datasets import Dataset
from src.model.causal_lm import CausalLanguageModel
from src.data.dataset_loader import tokenize_slice_batch, load_hf_dataset_with_sampling

from tqdm import tqdm

class LMfmap:
    def __init__(self, model: CausalLanguageModel, mode=['input', 'output'], device='cuda', fp16=True) -> None:
        # fmap
        self.fmap = None
        self.tokens_count = None

        # options
        self.mode = mode
        self.special_tracking = []

        # model: hugging face model
        self.model = model
        self.kn_act_buffer = model.enable_output_knowledge_neurons()
        self.vocab_list = model.get_vocab()
        self.n_layers = model.get_nb_layers()
        self.n_units = model.get_nb_knowledge_neurons(layer_id=0)

        # tensor format
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        # prepare fmap
        self.prepare_fmap()

    def prepare_fmap(self) -> None:
        """
        Prepare the fmap
        """

        self.fmap = {}
        vocab_size = len(self.vocab_list) 
        fmap_dim = (vocab_size, self.n_units)
        
        # init token/unit matrix
        '''
        fmap['input'][l][i][j]
        provides the accumulated activation of knowledge neuron j when vocab item i is the last token of the input
        '''
        if 'input' in self.mode:
            self.fmap['input'] = [torch.full(size=fmap_dim, fill_value=0.0, dtype=self.dtype) for l in range(self.n_layers)]
        if 'output' in self.mode:
            self.fmap['output'] = [torch.full(size=fmap_dim, fill_value=0.0, dtype=self.dtype) for l in range(self.n_layers)]
        self.tokens_count = {
            'input':torch.zeros(size=(vocab_size,)).int().to(self.device), 
            'output':torch.zeros(size=(vocab_size,)).int().to(self.device)}
        
    def add_position(self, window_size) -> None:
        logging.warning(f'[FMAP] Adding position tracking. Make sure that the window size is {window_size} during the fmap extraction.')
        fmap_pos_dim = (window_size, self.n_units)
        for mode in self.fmap:
            for l in range(self.fmap[mode]):
                fmap_pos = torch.full(size=fmap_pos_dim, fill_value=0.0, dtype=self.dtype)
                # Concat original fmap and fmap_pos
                self.fmap[mode][l] = torch.cat((self.fmap[mode][l], fmap_pos), dim=0)
        self.special_tracking.add('position')


    def update_token_unit(self, unit_tokens, kn_act, layer, unique_id, token_ids, tokens_count, old_token_count):
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
        index_mask = (expand_unique_id == token_ids.unsqueeze(-1)).t().type(self.dtype) 
        # compute the unit-token activations for the batch, on the device
        """
        index_mask: [n_uniques_in_sequence, n_tokens_in_sequence]. Line i has 1 where token i appear in the sequence
        kn_act[layer].view(kn_d1*kn_d2, kn_d3): [n_tokens_in_sequence, n_feats]

        batch_unit_token_cum: index_mask x kn_act[layer] = [n_uniques_in_sequence, n_feats].
        Line i correspond to the accumulated kn features obtained when token i is the input.
        """
        batch_unit_token_cum = torch.matmul(index_mask.to(self.device), kn_act[layer].to(self.device).view(kn_d1*kn_d2, kn_d3).type(self.dtype))
        # check nan and inf
        # if batch_unit_token_cum.isnan().any():
        #     logging.warning('[accumulated kn] nan detected!')
        # if batch_unit_token_cum.isinf().any():
        #     logging.warning('[accumulated kn] inf detected!')   
            
        # update the cumuluative average
        unit_tokens[layer][unique_id] = cumulative_average(
            new_item    = batch_unit_token_cum,
            new_count   = tokens_count[unique_id].unsqueeze(-1),
            old_count   = old_token_count[unique_id].unsqueeze(-1),
            old_average = unit_tokens[layer][unique_id.to(save_device)],
            device      = self.device,
        ).to(save_device)

        return unit_tokens


    def extract(self, dataset: Dataset, batch_size, window_size, window_stride):
        
        # preprocess dataset
        dataset_sliced_batched, n_batch = tokenize_slice_batch(dataset, self.model.tokenizer, batch_size, window_size, window_stride)

        # iterate on the dataset
        for input_ids, attention_mask in tqdm(dataset_sliced_batched, total=n_batch):
            d_batch, d_sent = input_ids.shape
            
            # forward pass
            output = self.model.forward_pass_nograd((input_ids, attention_mask), tokenize=False)
            
            # accumulate input and output token ids
            tokens_ids = {
                'input': input_ids.flatten().to(self.device).type(self.dtype),
                'output': torch.argmax(output.logits.detach(), dim=-1).flatten().to(self.device).type(self.dtype)
            }
            # count unique tokens
            unique_id = {}
            old_token_count = {}

            for mode in self.fmap.keys():
                """
                Detect the unique tokens_id in the input sequence, and count them.
                Update the total tokens_count.
                Save an old_tokens_count, used to compute the cumulative average
                """
                unique_tokens_with_count = torch.unique(tokens_ids[mode], return_counts=True)
                unique_id[mode] = unique_tokens_with_count[0].long() # set of unique tokens in the input (resp. output)
                count_id = unique_tokens_with_count[1] # corresponding count for each unique token
                old_token_count[mode] = self.tokens_count[mode].clone().detach() # save old count
                self.tokens_count[mode][unique_id[mode]] += count_id # update new count

            # for each layer accumulate the unit-token association
            for l in range(self.n_layers):
                # per token stats
                for mode in self.fmap.keys():
                    with torch.no_grad():
                        self.fmap[mode] = self.update_token_unit(
                            unit_tokens=self.fmap[mode],
                            kn_act=self.kn_act_buffer,
                            layer=l,
                            unique_id=unique_id[mode],
                            token_ids=tokens_ids[mode],
                            tokens_count=self.tokens_count[mode],
                            old_token_count=old_token_count[mode],)

        for mode in self.fmap.keys():  
            self.tokens_count[mode] = self.tokens_count[mode].cpu()
            for l in range(self.n_layers):
                # check nan and inf
                if self.fmap[mode][l].isnan().any():
                    logging.warning(f'[Finished][{mode}][{l}] nan detected!')
                if self.fmap[mode][l].isinf().any():
                    logging.warning(f'[Finished][{mode}][{l}] inf detected!')  
                self.fmap[mode][l] = self.fmap[mode][l].cpu()

        return self.fmap, self.tokens_count

def cumulative_average(new_item, new_count, old_count, old_average, device='cpu'):
    datatype = old_average.dtype
    new_item = new_item.to(device).type(torch.float32)
    new_count = new_count.to(device).type(torch.float32)
    old_count = old_count.to(device).type(torch.float32)
    old_average = old_average.to(device).type(torch.float32)
    old_sum = old_count * old_average # float32, to avoid overflow
    cum_avg = (new_item + (old_count) * old_average) / (new_count)
    # check nan and inf
    # if cum_avg.isnan().any():
    #     logging.warning('[cumulative average f32] nan detected!')
    # if cum_avg.isinf().any():
    #     logging.warning('[cumulative average f32] inf detected!')    
    cum_avg = cum_avg.type(datatype)
    # check nan and inf
    # if cum_avg.isnan().any():
    #     logging.warning('[cumulative average after cast] nan detected!')
    # if cum_avg.isinf().any():
    #     logging.warning('[cumulative average after cast] inf detected!')    
    return cum_avg
