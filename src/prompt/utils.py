import sys
sys.path.append('../')

def parse_paraphrases(filepath):
    template_dic = {}
    with open(filepath, 'r') as f:
        lines = f.read()
    relation = None
    for l in lines.split('\n'):
        if len(l)==0:
            continue
        elif l[0]=='*':
            relation = l[1:].split()[0]
            template_dic[relation]=[]
            continue
        template = l[3:]
        # only use template with the [Y] at the end because we are using causal LMs
        # remove [Y] because it is supposed to be always the last token
        if template.endswith('[Y]'):
            template_dic[relation].append(template)
        elif template[:-1].endswith('[Y]'):
            # it's ok if there is only one character (punctuation) after the [Y]. But we remove it.
            template_dic[relation].append(template[:-1])
    # remove duplicates
    for relation in template_dic:
        template_dic[relation] = set(template_dic[relation])
    
    
    return template_dic


