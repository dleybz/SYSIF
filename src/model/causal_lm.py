import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable


class CausalLanguageModel:
    def __init__(self, model_name, device="cpu", fast_tkn=True):
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tkn)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2)
        generated_text = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        return generated_text

    def forward_pass_from_text(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        output = self.model(input_ids)
        return output
    
    def forward_pass_from_tkns(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output

    def forward_pass_nograd(self, input, tokenize=True):
        with torch.no_grad():
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
            self.kn_act_buffer[layer] = torch.relu(output.detach()).cpu()
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
        return self.tokenizer.vocab

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

        

if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"  # You can replace this with the model name you want to use
    lm = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    print(lm)