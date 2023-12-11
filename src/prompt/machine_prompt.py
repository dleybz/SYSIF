import sys
sys.path.append('../')
import random
import string
import math
from src.prompt.utils import parse_paraphrases
from src.data.lama_dataset import LAMAset
from src.data.dataset_loader import batchify
from src.model.causal_lm import CausalLanguageModel
import random
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import statistics
from copy import deepcopy


class DiscreteGradientPromptSearch():
    def __init__(self, model: CausalLanguageModel, n_population, num_candidates,) -> None:
        self.model = model
        self.device = model.device
        self._stored_embeddings_gradient = None # the gradient will be store here
        self.prepare_model()
        self.num_candidates = num_candidates
        self.n_population = n_population
        self.temperature_norm=2e-1
        self.topk_display = 3
        self.n_rounds = 3
        self.p_flip = 0.4

    def prepare_model(self) -> None:
        """
        Register the hook to store embedding gradients
        """
        def hook(module, grad_in, grad_out):
            self._stored_embeddings_gradient = grad_out[0]
        self.model.get_embeddings().register_full_backward_hook(hook)

    def get_embedding_gradient(self,):
        return self._stored_embeddings_gradient

    def nll(self, predict_logits, label_ids):
        predict_logp = F.log_softmax(predict_logits, dim=-1)
        target_logp = predict_logp.gather(-1, label_ids.unsqueeze(-1))
        # target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
        # target_logp = torch.logsumexp(target_logp, dim=-1) # useless if only one label per example
        return -target_logp
    
    def preprocess_data(self, dataset):
        return dataset
    
    def temp_softmax(self, x, temperature, discard_zeros=False):
        if discard_zeros:
            x_temp = torch.where(x<1e-2, -1e9, x/temperature)
        else:
            x_temp = x/temperature
        return torch.softmax(x_temp, dim=0)

    """
    From Shin et al., 2020: https://arxiv.org/abs/2010.15980
    """
    def hotflip_attack(self, averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
        """Returns the top candidate replacements."""
        with torch.no_grad():
            gradient_dot_embedding_matrix = torch.matmul(
                embedding_matrix,
                averaged_grad
            )
            if filter is not None:
                gradient_dot_embedding_matrix -= filter
            if not increase_loss: # do you want to increase or decrease the loss?
                gradient_dot_embedding_matrix *= -1
            # sample from gradient dist
            score = gradient_dot_embedding_matrix.float()
            # score = score + score.min().abs() + 1e-9# only positive
            # score_normalized = score / score.sum() # normalize
            score_normalized = self.temp_softmax(score, temperature=self.temperature_norm)
            sampled_idx = torch.multinomial(score_normalized.cpu(), num_candidates, replacement=True).tolist()
        return sampled_idx
    
    def save(self, population_template, cpt_iteration, savepath):
        with open(savepath, 'a') as f_out:
            population_set = list(set(population_template))
            population_set.sort(reverse=True, key=lambda x:x[1])
            savelines = '\n'.join([f'{cpt_iteration}\t[START-TEMPLATE]{d[0]}[END-TEMPLATE]\t{d[1]:.2f}' for d in population_set])+'\n'
            f_out.write(savelines)

    def evaluate_candidates(self, template_candidates, lamaset, relation, batch_size, n_generated_tokens):
        # construct the prompts
        df_candidates = []
        for tid, this_template in enumerate(template_candidates):
            filled_list = lamaset.fill_template(relation, this_template, set='dev')
            df_temp = pd.DataFrame()
            df_temp['prompt'] = [tp[0] for tp in filled_list]
            df_temp['label'] = [tp[1] for tp in filled_list]
            # df_temp['relation'] = [relation,] * len(df_temp)
            df_temp['template'] = [this_template,] * len(df_temp)
            df_candidates.append(df_temp)
        df_candidates = pd.concat(df_candidates)
        # feed prompts to the LM and gather predictions
        prompt_list = df_candidates['prompt'].values.tolist()
        pred_list = []
        batches, n_batches = batchify(prompt_list, batch_size, drop_last=False, output_text=True)
        for batch in tqdm(batches, desc="[EVALUATION]"):
            text_generated = self.model.generate_tokens_from_text_batch(batch, n_tokens=n_generated_tokens)
            pred_list += text_generated
        df_candidates['pred'] = pred_list
        df_candidates['correct'] = df_candidates.apply(lambda row: row['label'] in row['pred'], axis=1)  
        return df_candidates
        

    def train(self, initial_population, lamaset, relation, n_iterations_max, batch_size, savepath):
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """

        # in the first iteration, the population size is > than self.n_population
        population_template = [(t, None) for t in initial_population]
        mem_template_info = {} # store the embedding gradient to avoid having to recompute it multiple time

        # first, eval the initial population
        df_eval = self.evaluate_candidates([t[0] for t in population_template if t[1] is None], lamaset, relation, batch_size, 2)
        population_template = [(d[0], d[1]) for d in df_eval.groupby('template')['correct'].mean().reset_index().values.tolist()]
        population_template.sort(reverse=True, key=lambda x:x[1])
        msg = '\n'.join([f'T:__{d[0]}__. S:{d[1]:.2f}' for d in population_template[:self.topk_display]])
        print(f'[INITIAL POPULATION]:\n'+msg)
        not_finished = True
        cpt_iteration = 0
        self.save(population_template, cpt_iteration, savepath)
        while(not_finished):
            cpt_iteration += 1

            for round in range(self.n_rounds):

                for (machine_template, template_score) in tqdm(deepcopy(population_template), desc=f"[TRAIN][it:{cpt_iteration}] Computing gradient for each template of the population",file=sys.stdout):
                    
                    if machine_template in mem_template_info:
                        averaged_template_gradient = mem_template_info[machine_template]['gradient']
                        tokenized_template, _ = lamaset.fill_template_and_tokenize(None, machine_template, self.model.tokenizer) # first arg to None to just get the tokenized template
                    else:
                        tokenized_template, filled_data = lamaset.fill_template_and_tokenize(relation, machine_template, self.model.tokenizer, set='train')
                        batches = [filled_data[i:i+batch_size] for i in range(0,len(filled_data),batch_size)]
                        accu_template_gradient = None
                        for batch in batches:
                            # prepare input
                            inputs = [torch.tensor(d[0]+d[1]) for d in batch]
                            labels = [torch.tensor(d[2]) for d in batch]
                            labels = [l[0] for l in labels] # only keep the first token. TODO: should we change that?
                            # tokenize and (right) pad the inputs
                            max_length = max([len(t) for t in inputs])
                            inputs = torch.stack([F.pad(t, (0, max_length-len(t)), value=self.model.tokenizer.pad_token_id) for t in inputs])
                            attention_mask = torch.where(inputs.eq(self.model.tokenizer.pad_token_id),0,1)
                            # todo: this is hacky
                            template_mask = torch.tensor([[0,]*len(d[0])+[1,]*len(d[1])+[0,]*(max_length-(len(d[0])+len(d[1]))) for d in batch]).bool()# 1 if the token is part of the template 0 otherwise
                            # feed the model with the data
                            output = self.model.forward_pass((inputs.to(self.device), attention_mask.to(self.device)), tokenize=False)
                            pred_id = attention_mask.sum(-1)-1 # be sure that padding is 'right'
                            pred_logit = output.logits[range(len(batch)), pred_id]
                            # compute loss
                            loss = self.nll(pred_logit, torch.tensor(labels).to(self.device)).mean()
                            # compute gradient of loss vs input embedding
                            loss.backward()
                            embeddings = self.model.get_embeddings().weight
                            embeddings_gradient = self.get_embedding_gradient()
                            # only keep the gradient of the template tokens
                            template_gradient = torch.masked_select(embeddings_gradient, template_mask.unsqueeze(-1).to(self.device)).view(len(batch), len(tokenized_template), embeddings.size(-1))
                            accu_template_gradient = (accu_template_gradient + template_gradient.sum(0)) if accu_template_gradient is not None else template_gradient.sum(0)
                        averaged_template_gradient = accu_template_gradient / len(filled_data)
                        # save the embedding gradient for later
                        mem_template_info[machine_template] = {'gradient': averaged_template_gradient.detach().clone(), 'score': template_score}
                    # Mutation: hotflip attack (from Autoprompt)
                    with torch.no_grad():
                        len_tokenized_template = len(tokenized_template)
                        for idx_tkn in range(len_tokenized_template):
                            p = random.random()
                            if p < (self.p_flip/(2**round)):
                                sampled_tokens = self.hotflip_attack(averaged_template_gradient[idx_tkn], embeddings, num_candidates=self.num_candidates)
                                # Add mutated templates to the population
                                for token_candidate in sampled_tokens:
                                    temp = tokenized_template.copy()
                                    temp[idx_tkn] = token_candidate
                                    try:
                                        temp_text = '[X] '+self.model.tokenizer.decode(temp)+ ' [Y]'
                                        # check if we already know the score of the mutated template
                                        if temp_text in mem_template_info:
                                            temp_score = mem_template_info[temp_text]['score']
                                        else:
                                            temp_score = None
                                        population_template.append((temp_text, temp_score)) # (text_template, score)
                                    except TypeError: # can happens if something goes wrong with the tokenizer
                                        continue # skip it
            
            # remove dupplicate, but keep track of them
            population_template_undup = []
            population_template_undup_count = {}
            for t in population_template:
                if t not in population_template_undup:
                    population_template_undup.append(deepcopy(t))
                    population_template_undup_count[t[0]] = 1
                else: # dupplicate
                    population_template_undup_count[t[0]] += 1
            population_template = population_template_undup
            # evaluate the new templates in the population
            df_eval = self.evaluate_candidates([t[0] for t in population_template if t[1] is None], lamaset, relation, batch_size, 1)
            population_template = [(d[0], d[1]) for d in df_eval.groupby('template')['correct'].mean().reset_index().values.tolist()]\
                                + [t for t in population_template if t[1] is not None]
            # redupplicate
            population_template_redup = []
            for t in population_template:
                population_template_redup += [deepcopy(t),]*population_template_undup_count[t[0]]
            population_template = population_template_redup

            # select the best template of the population (sampling)
            scores = torch.tensor([d[1] for d in population_template]) #+ 1e-9
            score_normalized = self.temp_softmax(scores, temperature=self.temperature_norm, discard_zeros=True)
            sampled_idx = torch.multinomial(score_normalized, self.n_population, replacement=True).tolist()
            population_template = [population_template[i] for i in sampled_idx]
            population_template.sort(reverse=True, key=lambda x:x[1])
            # print
            msg = '\n'.join([f'T:__{d[0]}__. S:{d[1]:.2f}' for d in population_template[:self.topk_display]])
            print(f'[i-{cpt_iteration}]:\n'+msg)
            # save templates
            self.save(population_template, cpt_iteration, savepath)
            # stop training?
            not_finished = (cpt_iteration <= n_iterations_max)
            # release memory, if a template has a score lower or equal to the median of the current population
            if cpt_iteration%10==0:
                med_score = statistics.median([s[1] for s in population_template])
                for t in list(mem_template_info.keys()):
                    if mem_template_info[t]['score'] <= med_score:
                        del mem_template_info[t]

        return machine_template



class EvoMachinePrompt():
    def __init__(self, mutate_function, crossover_function, fitness_function) -> None:
        RELATION='P1001'
        TEMPLATES='data/paraphrases/relation-paraphrases_v2.txt'

        paraphrases=parse_paraphrases(TEMPLATES)

        # Define the population sizen
        self.population_size = 100

        # Define the number of generations
        self.num_generations = 50

        # Define the mutation rate
        self.mutation_rate = 0.1

        # define evolution functions
        self.fitness_function = fitness_function
        self.mutate = mutate_function 
        self.crossover = crossover_function

        # Initialize the population with human paraphrases
        self.population=[] 
        self.population += paraphrases[RELATION]
        self.population += random.choices(list(paraphrases[RELATION]), k=(self.population_size-len(self.population)))# extend the population to reach the population size by dupplicating elements

    def evolution(self):
        # Main loop for generations
        for generation in range(self.num_generations):
            # Evaluate the fitness of each sentence in the population / here the fitness is the LAMA evaluation
            # fitness_scores = [self.fitness_function(sentence, target_sentence) for sentence in population]

            # Select the top-performing sentences
            # top_performers = [sentence for _, sentence in sorted(zip(fitness_scores, population), reverse=True)[:population_size // 2]]

            # Generate new sentences through mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(self.population, 2)
                child = self.mutate(self.crossover(parent1, parent2))
                new_population.append(child)
            self.population = random.sample(self.population+new_population, self.population_size)

            # correcteur orthographique

            # for _ in range(self.population_size - len(top_performers)):
            #     parent1, parent2 = random.sample(top_performers, 2)
            #     child = mutate(crossover(parent1, parent2))
            #     new_population.append(child)

            # # Replace the old population with the new population
            # population = top_performers + new_population

            # # Optional: Check if the target sentence has been found
            # if target_sentence in population:
            #     print(f"Found target sentence in generation {generation + 1}!")
            #     break

        print(self.population)

# Function to calculate fitness
def init_lama_fitness(lamaset: LAMAset):

    def fitness_function(template):
        fitness_score = 0
        return fitness_score
    
    return fitness_function

# Function to perform mutation
def init_char_mutate(p_rand=0.05, p_suppr=0.05, p_dupp=0.05, subject_token='[X]'):

    def mutate_function(sentence):
        mutated_sentence = ''
        parsed_sentence = my_char_parser(sentence, subject_token)
        for c in parsed_sentence:
            mutated_sentence += mutate_char(c, p_rand, p_suppr, p_dupp)
        return mutated_sentence

    return mutate_function

def mutate_char(c, p_rand, p_suppr, p_dupp):
    max_utf8 = 1114111
    p = random.random()
    if p < p_suppr: # suppr character
        mutated_c = ''
    elif p < (p_suppr+p_dupp): # dupplicate character
        mutated_c = c+c
    elif p < (p_suppr+p_dupp+p_rand): # random replace
        mutated_c = chr(random.randint(0,max_utf8))
    else: # no mutation
        mutated_c = c
    return mutated_c

# Function to perform crossover
def init_char_crossover(subject_token='[X]'):
    # Implement a crossover operation to generate a new sentence from two parents.
    # For example, you can choose a random point to split the parents and combine their segments.

    def crossover_function(sent1, sent2):
        parsed_sent1 = my_char_parser(sent1, subject_token)
        parsed_sent2 = my_char_parser(sent2, subject_token)
        # id_subj_1 = parsed_sent1.index(subject_token) if subject_token in parsed_sent1 else None
        # id_subj_2 = parsed_sent2.index(subject_token) if subject_token in parsed_sent2 else None
        t1 = random.randint(1, len(parsed_sent1) - 1)
        t2 = random.randint(1, len(parsed_sent2) - 1)
        child = ''.join(parsed_sent1[:t1]+parsed_sent2[t2:])
        return child

    return crossover_function

def my_char_parser(sentence, subject_token='[X]'):
    c_i = 0
    char_list = []
    while(c_i<len(sentence)):
        if sentence[c_i:c_i+3]==subject_token:
            c_i += 3
            c = subject_token
        else:
            c = sentence[c_i]
            c_i += 1
        char_list.append(c)
    return char_list