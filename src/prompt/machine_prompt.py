import sys
sys.path.append('../')
import random
import string
from src.prompt.utils import parse_paraphrases
from src.data.lama_dataset import LAMAset
from src.data.dataset_loader import batchify
from src.model.causal_lm import CausalLanguageModel
import random
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


class DiscreteGradientPromptSearch():
    def __init__(self, model: CausalLanguageModel,) -> None:
        self.model = model
        self.device = model.device
        self._stored_embeddings_gradient = None # the gradient will be store here
        self.prepare_model()
        self.n_generated_tokens = 2
        self.num_candidates = 5
        self.n_population = 5

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
            _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

        return top_k_ids

    def evaluate_candidates(self, template_candidates, lamaset, relation, batch_size):
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
            text_generated = self.model.generate_tokens_from_text_batch(batch, n_tokens=self.n_generated_tokens)
            pred_list += text_generated
        df_candidates['pred'] = pred_list
        df_candidates['correct'] = df_candidates.apply(lambda row: row['label'] in row['pred'], axis=1)  
        return df_candidates
        

    def train(self, initial_population, lamaset, relation, n_iterations_max, batch_size):
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """

        # in the first iteration, the population size is > than self.n_population
        population_template = [(t, None) for t in initial_population]

        # first, eval the initial population
        df_eval = self.evaluate_candidates([t[0] for t in population_template if t[1] is None], lamaset, relation, batch_size)
        print(df_eval)
        population_template = [(d[0], d[1]) for d in df_eval.groupby('template')['correct'].mean().reset_index().values.tolist()]
        msg = '\n'.join([f'T:__{d[0]}__. S:{d[1]}' for d in population_template])
        print(f'[INITIAL POPULATION]:'+msg)

        not_finished = True
        cpt_iteration = 0
        while(not_finished):
            cpt_iteration += 1

            for (machine_template, _) in tqdm(population_template.copy(), desc=f"[TRAIN][it:{cpt_iteration}] Computing gradient for each template of the population",file=sys.stdout):
                
                filled_data = lamaset.fill_template_and_tokenize(relation, machine_template, self.model.tokenizer, set='train')
                batches = [filled_data[i:i+batch_size] for i in range(0,len(filled_data),batch_size)]
                tokenized_template = batches[0][0][1] # check that it is equal to batch[1][1]
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
                    template_gradient = torch.masked_select(embeddings_gradient, template_mask.unsqueeze(-1).to(self.device)).view(len(batch), -1, embeddings.size(-1))
                    accu_template_gradient = (accu_template_gradient + template_gradient.sum(0)) if accu_template_gradient is not None else template_gradient.sum(0)
                # Mutation: hotflip attack (from Autoprompt)
                with torch.no_grad():
                    # randomly pick a token which will be mutated
                    averaged_template_gradient = accu_template_gradient / len(filled_data)
                    token_to_mutate = random.randrange(averaged_template_gradient.size(0))
                    top_k_ids = self.hotflip_attack(averaged_template_gradient[token_to_mutate], embeddings, num_candidates=self.num_candidates)
                # Add mutated templates to the population
                for token_candidate in top_k_ids:
                    temp = tokenized_template.copy()
                    temp[token_to_mutate] = token_candidate.item()
                    temp_text = '[X] '+self.model.tokenizer.decode(temp) + ' [Y]'
                    population_template.append((temp_text, None)) # (text_template, score)
            
            # evaluate the new templates in the population
            df_eval = self.evaluate_candidates([t[0] for t in population_template if t[1] is None], lamaset, relation, batch_size)
            population_template = [(d[0], d[1]) for d in df_eval.groupby('template')['correct'].mean().reset_index().values.tolist()]\
                                + [t for t in population_template if t[1] is not None]

            # select the best template of the population (sampling)
            scores = torch.tensor([d[1] for d in population_template])+ 1e-9
            norm_scores = scores / scores.sum()
            sampled_idx = torch.multinomial(norm_scores, self.n_population, replacement=True).tolist()
            population_template = [population_template[i] for i in sampled_idx]
            # print
            # if cpt_iteration%100==1:
            msg = '\n'.join([f'T:__{d[0]}__. S:{d[1]}' for d in population_template])
            print(f'[i-{cpt_iteration}]:'+msg)
            # stop training?
            not_finished = (cpt_iteration <= n_iterations_max)
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