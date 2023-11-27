import torch
import torch.nn.functional as F
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable


class CausalLanguageModel:
    def __init__(self, model_name, device="cpu", fast_tkn=True, fp16=True, padding_side='right'):
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = self.prepare_tokenizer(model_name, fast_tkn, padding_side=padding_side)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if fp16 else torch.float32).to(self.device)
        # print(self.model)
        self.layer_act = self.get_act_fn()

    def generate_tokens_batch(self, tkns_input, n_tokens):
        tokens_generated = self.model.generate(
            input_ids=tkns_input.input_ids,
            attention_mask=tkns_input.attention_mask,
            max_new_tokens=n_tokens, do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        tokens_generated = tokens_generated[:, -n_tokens:]
        return tokens_generated

    def generate_tokens_from_text_batch(self, text_input, n_tokens):
        input_ids = self.tokenizer(text_input, padding=True, return_tensors='pt').to(self.device)
        tokens_generated = self.generate_tokens_batch(input_ids, n_tokens)
        text_generated = [self.tokenizer.decode(t) for t in tokens_generated]
        return text_generated
    
    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2)
        generated_text = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        return generated_text

    def forward_per_layer(self, inputs):
        n_layers = self.get_nb_layers()
        for l_idx in range(n_layers):
            # process
            print('')
        return None

    def forward_pass_from_text(self, input_text):
        input_ids = self.tokenizer(input_text, padding=True, return_tensors='pt').to(self.device)
        output = self.model(**input_ids)
        return (output, input_ids['attention_mask'])
    
    def forward_pass_from_tkns(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output

    def forward_pass_nograd(self, input, tokenize=True):
        with torch.no_grad():
            output = self.forward_pass(input, tokenize)    
        return output
    
    def forward_pass(self, input, tokenize=True):
        if tokenize:
            output = self.forward_pass_from_text(input)
        else:
            input_ids, attention_mask = input
            output = self.forward_pass_from_tkns(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        return output

    def get_output_probability(self, input_text):
        output = self.forward_pass_nograd(input_text)
        logits = output.logits
        probabilities = logits.softmax(dim=-1)
        return probabilities
    
    def enable_output_hidden_states(self):
        self.model.config.output_hidden_states=True

    def enable_output_attention_maps(self):
        self.model.config.output_attentions=True

    def enable_output_knowledge_neurons(self):
        # define the tensor where the activation will be stored
        layers = self.get_layers()
        n_layers = len(layers)
        self.kn_act_buffer={lid: torch.empty(0) for lid in range(n_layers)}
        # Setting a hook for saving FFN intermediate output
        for lid, layer in enumerate(layers):
            self.get_knowledge_neurons(lid).register_forward_hook(self.save_kn_act_hook(lid))
        return self.kn_act_buffer

    def save_kn_act_hook(self, layer) -> Callable:
        def fn(_, __, output):
            before_act = output.detach()
            after_act = self.layer_act(before_act) # apply activation
            self.kn_act_buffer[layer] = after_act.cpu()
        return fn   

    # def get_attention_maps(self, input_text):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_ids)
    #         attentions = output.attentions
    #     return attentions

    # def get_activation_values(self, input_text):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_ids)
    #         hidden_states = output.hidden_states
    #     return hidden_states

    # def get_intermediate_representation(self, input_text, layer_id):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_ids)
    #         intermediate_layer = output.hidden_states[layer_id]
    #     return intermediate_layer 
    
    def get_vocab(self):
        if 'opt' in self.model_name:
            vocab = list(self.tokenizer.encoder)
        elif 'pythia' in self.model_name:
            vocab = self.tokenizer.vocab
        else:
            vocab = self.tokenizer.vocab
        return vocab
    
    def get_vocab_size(self):
        # warning: tokenizer.vocab_size only return the vocab size without taking into account the added tokens.
        return len(self.tokenizer)

    def get_embeddings(self):
        if 'opt' in self.model_name:
            embeddings = self.model.model.decoder.embed_tokens
        elif 'pythia' in self.model_name:
            embeddings = self.model.gpt_neox.embed_in
        else:
            embeddings = self.model.gpt_neox.embed_in
        return embeddings

    def get_layers(self):
        if 'opt' in self.model_name:
            layers = self.model.model.decoder.layers
        elif 'pythia' in self.model_name:
            layers = self.model.gpt_neox.layers 
        else:
            layers = self.model.gpt_neox.layers
        return layers

    def get_nb_layers(self):
        return len(self.get_layers())
    
    def get_knowledge_neurons(self, layer_id):
        layers = self.get_layers()
        if 'opt' in self.model_name:
            kn = layers[layer_id].fc1
        elif 'pythia' in self.model_name:
            kn = layers[layer_id].mlp.dense_h_to_4h
        else:
            kn = layers[layer_id].fc1
        return kn
    
    def get_nb_knowledge_neurons(self, layer_id=None):
        if layer_id is not None:
            return self.get_knowledge_neurons(layer_id).out_features
        else:
            return sum([self.get_knowledge_neurons(i).out_features for i in range(self.get_nb_layers())])

    def get_act_fn(self):
        if 'opt' in self.model_name:
            act_str = self.model.model.config.activation_function
        elif 'pythia' in self.model_name:
            act_str = self.model.config.hidden_act
        else: # default
            act_str = 'relu'
        
        if act_str == 'relu':
            return torch.nn.ReLU()
        elif act_str == 'gelu':
            return torch.nn.GELU()
        else: # relu by default
            return torch.nn.GELU()


    def prepare_tokenizer(self, model_name, fast_tkn, padding_side):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tkn, padding_side=padding_side)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        return tokenizer

    def prompt(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.model.generate(**inputs.to(self.device))
        return self.tokenizer.decode(tokens[0])

if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"  # You can replace this with the model name you want to use
    lm = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    print(lm)