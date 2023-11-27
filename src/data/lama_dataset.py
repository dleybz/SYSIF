import sys
sys.path.append('../')
import logging
import os
import json
from datasets import load_dataset, Dataset
import pandas as pd

class LAMAset:
    def __init__(self, lama_path, portion=1) -> None:

        self.dataset, self.info = load_lama_local(lama_path)
        self.dataset = self.dataset.sample(int(len(self.dataset)*portion))
    
    def lamaset_per_relation(self, relation):
        dataset_rel = self.dataset[self.dataset['predicate_id']==relation]
        info_rel = self.info[self.info['relation']==relation]

        return dataset_rel, info_rel

    def preprocess(self, balance=True, no_overlap=True):
        """
        train/val/test split
        """
        # balance data (optional) | only the test?
        # reorganise train/set to remove answer overlap

        return None

    def fill_template(self, relation, template, set='train'):
        """
        return a list of tuple (template(object), subject)

        A template should be in the form "[X] ..... [Y]"
        But [X] is not necessarly at the beginning.

        Return a list of tuples: (filled template, object)
        """
        if not template.endswith('[Y]'):
            logging.warning(f'[LAMA] Trying to fill in a template that doesnt end with [Y] -> STOP\n{template}')
            return None
        else:
            # troncate the template by removing [Y]
            template = template[:-3]

        this_set = self.dataset[self.dataset['set']==set]
        this_set = this_set[this_set['predicate_id']==relation]
        pair_list = this_set[['sub_label', 'obj_label']].values.tolist()
        
        filled_data = [(template.replace('[X]', subj).strip(), obj) for subj, obj in pair_list]
        return filled_data

    def fill_template_and_tokenize(self, relation, template, tokenizer, set='train'):
        """
        Fill and tokenize the templates

        A template should be in the form "[X] ..... [Y]"
        But [X] is not necessarly at the beginning.

        Return a list of tuples: (tokn_obj, tokn_core_template, tkn_object)
        """
        if not template.endswith('[Y]'):
            logging.warning(f'[LAMA] Trying to fill in a template that doesnt end with [Y] -> STOP\n{template}')
            return None
        else:
            # troncate the template by removing [Y]
            template = template[:-3]

        this_set = self.dataset[self.dataset['set']==set]
        this_set = this_set[this_set['predicate_id']==relation]
        pair_list = this_set[['sub_label', 'obj_label']].values.tolist()
        
        core_tokenized = tokenizer.encode(template.replace('[X]', '').strip())
        # TODO: adding the space before obj:-> this is very Pythia specific, change it
        filled_data = [(tokenizer.encode(subj), core_tokenized, tokenizer.encode(' '+obj)) for subj, obj in pair_list]
        return filled_data
    
    def evaluate(self):
        return None
    
    def get_relations(self, set='test'):
        return list(self.dataset[self.dataset['set']==set]['predicate_id'].unique())

def load_lama_local(datapath):
    """
    Load the lama dataset stored in a local directory.
    Return a pandas dataframe
    """
    lama_dataset = []
    relation_folders = [f[0].split('/')[-1] for f in os.walk(datapath)]
    relation_folders = [f for f in relation_folders if (len(f)>0 and f[0]=='P')]
    for rel_f in relation_folders:
        # rel_f is a folder containing the dev/test/train jsonl
        for split in ['dev', 'test', 'train']:
            df = pd.read_json(path_or_buf=os.path.join(datapath, rel_f, split+'.jsonl'), lines=True)
            df = df[["obj_label",  "sub_label", "predicate_id"]]
            df['set'] = [split,]*len(df)
            lama_dataset.append(df)
    lama_dataset = pd.concat(lama_dataset)
    lama_info = pd.read_json(path_or_buf=os.path.join(datapath, 'LAMA_relations.jsonl'), lines=True)
    return lama_dataset, lama_info