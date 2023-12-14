import sys
sys.path.append('../')
import random
import string
import math
import logging
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
    def __init__(self, model: CausalLanguageModel, n_population, num_candidates, n_rounds) -> None:
        self.model = model
        self.device = model.device
        self._stored_embeddings_gradient = None # the gradient will be store here
        self.prepare_model()
        self.num_candidates = num_candidates
        self.n_population = n_population
        self.temperature_norm=1e-2
        self.topk_display = 3
        self.n_rounds = n_rounds
        self.p_flip = 0.4
        # memory
        self.mem_template_info = {} # store the embedding gradient to avoid having to recompute it multiple time


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
                   filter=None,
                   replacement=True):
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
            sampled_idx = torch.multinomial(score_normalized.cpu(), num_candidates, replacement=replacement).tolist()
        return sampled_idx
    
    def save(self, population_template, cpt_iteration, savepath):
        with open(savepath, 'a') as f_out:
            population_set = list(set(population_template))
            population_set.sort(reverse=True, key=lambda x:x[1])
            savelines = '\n'.join([f'{cpt_iteration}\t{d[2]}\t[START-TEMPLATE]{d[0]}[END-TEMPLATE]\t{d[1]:.2f}' for d in population_set])+'\n'
            f_out.write(savelines)

    def deduplicate_templates(template_candidates):
        population_template_undup = []
        population_template_undup_count = {}
        for t in template_candidates:
            if t not in population_template_undup:
                population_template_undup.append(deepcopy(t))
                population_template_undup_count[t[0]] = 1
            else: # dupplicate
                population_template_undup_count[t[0]] += 1
        template_candidates = population_template_undup
        return template_candidates, population_template_undup_count
    
    def deduplicate_templates(self, template_candidates, population_template_undup_count):
        population_template_redup = []
        for t in population_template:
            population_template_redup += [deepcopy(t),]*population_template_undup_count[t[0]]
        population_template = population_template_redup
        return population_template

    def evaluate_candidates(self, template_candidates, lamaset, relation, batch_size, n_generated_tokens, return_pred=False):
        # remove dupplicates, but keep track of them
        template_candidates, population_template_undup_count = self.deduplicate_templates(template_candidates)
        # select those which have to be evaluated
        template_to_evaluate = [(t[0], t[2]) for t in template_candidates if t[1] is None]
        # construct the prompts
        df_candidates = []
        for (this_template, tid) in template_to_evaluate:
            filled_list = lamaset.fill_template(relation, this_template, set='dev')
            df_temp = pd.DataFrame()
            df_temp['prompt'] = [tp[0] for tp in filled_list]
            df_temp['label'] = [tp[1] for tp in filled_list]
            df_temp['tid'] = [tid,] * len(df_temp)
            # df_temp['relation'] = [relation,] * len(df_temp)
            df_temp['template'] = [this_template,] * len(df_temp)
            df_candidates.append(df_temp)
        df_candidates = pd.concat(df_candidates)
        # foward pass: feed prompts to the LM and gather predictions
        prompt_list = df_candidates['prompt'].values.tolist()
        pred_list = []
        batches, n_batches = batchify(prompt_list, batch_size, drop_last=False, output_text=True)
        for batch in tqdm(batches, desc="[EVALUATION]"):
            text_generated = self.model.generate_tokens_from_text_batch(batch, n_tokens=n_generated_tokens)
            pred_list += text_generated
        df_candidates['pred'] = pred_list
        # evaluate
        df_candidates['correct'] = df_candidates.apply(lambda row: row['label'] in row['pred'], axis=1)  

        # todo: use a more balanced metric like below
        # result = df_candidates.groupby(['template','tid','label']).mean('correct').groupby(['template','tid']).mean().reset_index().values.tolist()

        population_template = [(d[0], d[2], d[1]) for d in df_candidates.groupby(['template','tid'])['correct'].mean().reset_index().values.tolist()]\
                                + [t for t in template_candidates if t[1] is not None]
        # redupplicate
        population_template = self.deduplicate_templates(template_candidates, population_template_undup_count)
        
        if return_pred:
            return population_template, df_candidates
        else:
            return population_template
        
    def select_candidates(self, population_template):
        scores = torch.tensor([d[1] for d in population_template]) #+ 1e-9
        score_normalized = self.temp_softmax(scores, temperature=self.temperature_norm, discard_zeros=True)
        sampled_idx = torch.multinomial(score_normalized, self.n_population, replacement=True).tolist()
        population_template = [population_template[i] for i in sampled_idx]
        return population_template

    def print_population(self, population_template):
        population_template.sort(reverse=True, key=lambda x:x[1])
        msg = '\n'.join([f'[{d[2]}] T:__{d[0]}__. S:{d[1]:.2f}' for d in population_template[:self.topk_display]])
        print(f'[INITIAL POPULATION]:\n'+msg)


    def new_generation(self, population_template, p_flip, lamaset, relation, batch_size, cpt_iteration, template_count):
        for (machine_template, template_score, tid) in tqdm(deepcopy(population_template), desc=f"[TRAIN][it:{cpt_iteration}] Computing gradient for each template of the population",file=sys.stdout):
            if tid in self.mem_template_info:
                averaged_template_gradient = self.mem_template_info[tid]['gradient']
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
                self.mem_template_info[tid] = {'gradient': averaged_template_gradient.detach().clone(), 'score': template_score}
            # Mutation: hotflip attack (from Autoprompt)
            with torch.no_grad():
                len_tokenized_template = len(tokenized_template)
                for idx_tkn in range(len_tokenized_template):
                    p = random.random()
                    if p < p_flip:
                        sampled_tokens = self.hotflip_attack(averaged_template_gradient[idx_tkn], embeddings, num_candidates=self.num_candidates)
                        # Add mutated templates to the population
                        for token_candidate in sampled_tokens:
                            temp = tokenized_template.copy()
                            temp[idx_tkn] = token_candidate
                            try:
                                temp_text = '[X] '+self.model.tokenizer.decode(temp)+ ' [Y]'
                                # check if we already know the score of the mutated template
                                if temp_text in self.mem_template_info:
                                    temp_score = self.mem_template_info[temp_text]['score']
                                else:
                                    temp_score = None
                                template_count += 1 # increase template count
                                population_template.append((temp_text, temp_score, f'{tid}-{template_count}')) # (text_template, score)
                            except TypeError: # can happens if something goes wrong with the tokenizer
                                continue # skip it
        return population_template, template_count
    
    def evolution_step(self, population_template, n_rounds, lamaset, relation, batch_size, cpt_iteration, template_count):
        for round in range(n_rounds):
            population_template, template_count = self.new_generation(self, population_template, self.p_flip/(2**round), lamaset, relation, batch_size, cpt_iteration, template_count)
            
        # evaluate the new templates in the population
        population_template = self.evaluate_candidates(population_template, lamaset, relation, batch_size, 1)
        # select the best template of the population (sampling)
        population_template = self.select_candidates(population_template)
        return population_template, template_count

    def train(self, initial_population, lamaset, relation, n_iterations_max, batch_size, savepath):
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """

        # in the first iteration, the population size is > than self.n_population
        # (template, score, template_id)
        # the template_id is constructing by concatenanting the parent id + a new number
        # here the parent id is R (root).
        population_template = [(t, None, f'R-{t_id}') for t_id, t in enumerate(initial_population)]

        # first, eval the initial population
        population_template = self.evaluate_candidates(population_template, lamaset, relation, batch_size, 2)
        self.print_population(population_template)

        not_finished = True
        cpt_iteration = 0
        self.save(population_template, cpt_iteration, savepath)
        # template count
        tcpt = len(population_template)

        while(not_finished):
            cpt_iteration += 1

            population_template, tcpt = self.evolution_step(self, population_template, self.n_rounds, lamaset, relation, batch_size, cpt_iteration, tcpt)

            self.print_population(population_template)
            # save templates
            self.save(population_template, cpt_iteration, savepath)
            # stop training?
            not_finished = (cpt_iteration <= n_iterations_max)

            # release memory, if a template has a score lower or equal to the median of the current population
            if cpt_iteration%10==0:
                med_score = statistics.median([s[1] for s in population_template if s[1] is not None])
                for t in list(self.mem_template_info.keys()):
                    # hacky: control why we sometimes have none
                    if self.mem_template_info[t]['score'] is None or self.mem_template_info[t]['score'] <= med_score:
                        del self.mem_template_info[t]

        return population_template

class OneTokenGradientPromptSearch(DiscreteGradientPromptSearch):
    def __init__(self, model: CausalLanguageModel, n_population, num_candidates, mode) -> None:
        super().__init__(self, model, 1, num_candidates, n_rounds=1)
        self.mode = mode # hot, neutral

    def search(self, human_templates, savepath, lamaset, relation, batch_size):
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """

        # Initialise the population with human templates
        human_templates = [(t, None, f'R-{t_id}') for t_id, t in enumerate(human_templates)]

        # first, eval the initial population
        human_templates, df_human = self.evaluate_candidates(human_templates, lamaset, relation, batch_size, 2, return_pred=True)
        
        # filter to only keep template with accuracy > 10
        human_templates = [h for h in human_templates if h[1]>0.1]
        
        self.print_population(human_templates)

        if len(human_templates)==0:
            logging.warning('[One token search] No human template is accurate enough. Stop.')
            return None

        # template count
        tcpt = len(human_templates)

        for (human, h_score, h_tid) in human_templates:

            self.save([(human, h_score, h_tid),], h_tid, savepath)

            tokenized_template, filled_data = lamaset.fill_template_and_tokenize(relation, human, self.model.tokenizer, set='train')
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

                if self.mode=='hot':
                    targets = torch.tensor(labels).to(self.device)
                elif self.mode=='neutral':
                    targets = torch.argmax(pred_logit, dim=-1) # model's predictions

                # compute loss
                loss = self.nll(pred_logit, targets).mean()
                # compute gradient of loss vs input embedding
                loss.backward()
                embeddings = self.model.get_embeddings().weight
                embeddings_gradient = self.get_embedding_gradient()
                # only keep the gradient of the template tokens
                template_gradient = torch.masked_select(embeddings_gradient, template_mask.unsqueeze(-1).to(self.device)).view(len(batch), len(tokenized_template), embeddings.size(-1))
                accu_template_gradient = (accu_template_gradient + template_gradient.sum(0)) if accu_template_gradient is not None else template_gradient.sum(0)
            averaged_template_gradient = accu_template_gradient / len(filled_data)
            
            # Mutation: hotflip attack (from Autoprompt)
            candidates_templates = []
            with torch.no_grad():
                len_tokenized_template = len(tokenized_template)
                for idx_tkn in range(len_tokenized_template):
                    sampled_tokens = self.hotflip_attack(averaged_template_gradient[idx_tkn], embeddings, num_candidates=self.num_candidates, replacement=False)
                    # Add mutated templates to the population
                    for token_candidate in sampled_tokens:
                        temp = tokenized_template.copy()
                        temp[idx_tkn] = token_candidate
                        try:
                            temp_text = '[X] '+self.model.tokenizer.decode(temp)+ ' [Y]'
                            template_count += 1 # increase template count
                            candidates_templates.append((temp_text, None, f'{h_tid}-{template_count}')) # (text_template, score)
                        except TypeError: # can happens if something goes wrong with the tokenizer
                            continue # skip it

            candidates_templates, df_candidates = self.evaluate_candidates(self, candidates_templates, lamaset, relation, batch_size, 2, return_red=True)

            # compare each candidate's predictions with the human template prediction:

            # print and save
            self.print_population(candidates_templates)
            self.save(candidates_templates, h_tid, savepath)

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