from collections import defaultdict
import itertools
import json
import random
import re
import sys



from tqdm import tqdm
from utils import load_jsonl
from construct_syn_kgqa.llm import run_llm


def convert_list_to_str(list_, sep=", "):
    return "[" + sep.join(f'"{w}"' for w in list_) + "]"


data_filepath = "data/ComplexWebQuestions/ComplexWebQuestions_dev.bound_triples.jsonl"

samples = load_jsonl(data_filepath)

relation_to_cnt = defaultdict(int)
relation_to_head = defaultdict(set)
relation_to_tail = defaultdict(set)
hop_to_cnt = defaultdict(int)


for sample in samples:
    all_triples = list(itertools.chain(*sample["all_bound_triples"]))
    if len(all_triples):
        hop_to_cnt[len(sample["all_bound_triples"][0])] += 1

    for triple in all_triples:
        relation = triple[1]
        relation_to_cnt[relation] += 1
        relation_to_head[relation].add(triple[0])
        relation_to_tail[relation].add(triple[-1])

relation_to_cnt = dict(sorted(relation_to_cnt.items(), key=lambda x: x[1], reverse=True))

hop_to_cnt = dict(sorted(hop_to_cnt.items(), key=lambda x: x[1], reverse=True))


def is_cvt_node(ent):
    if ent[:2] in ("m.", "g."):
        return True
    else:
        return False


remained_relation_to_head = defaultdict(set)
remained_relation_to_tail = defaultdict(set)

for rel in relation_to_cnt.keys():
    ents = relation_to_tail[rel]
    cvt_nodes = [is_cvt_node(ent) for ent in ents]
    if not any(cvt_nodes):
        remained_relation_to_tail[rel] = ents

    ents = relation_to_head[rel]
    cvt_nodes = [is_cvt_node(ent) for ent in ents]
    if not any(cvt_nodes):
        remained_relation_to_head[rel] = ents

with open("./construct_syn_kgqa/prompts", "r") as f:
    prompt = f.read().rstrip("\n")


relation_to_syn_head = defaultdict(set)
relation_to_syn_tail = defaultdict(set)


def extract_list(output):
    pattern = r"\[(.*?)\]"
    result = re.findall(pattern, output)
    if result:
        return result[0]
    else:
        raise NotImplementedError


def get_syn_ents(relation, ents, n_ents=10):
    ents = random.sample(list(ents), min(10, len(ents)))
    ents = convert_list_to_str(ents)

    pattern = "{prompt}\n\nRelation: {relation}\nents: {ents}\nOutput: "
    input = pattern.format(prompt=prompt, relation=relation, ents=ents)

    while True:
        try:
            output = run_llm(input)
            print(output)
            
            syn_ents = eval(extract_list(output))
            
            break
        except Exception as e:
            print(e)

    return syn_ents


for rel, ents in tqdm(list(remained_relation_to_head.items())):
    print(rel)
    syn_ents = get_syn_ents(rel, ents)
    relation_to_syn_head[rel] = syn_ents

for rel, ents in tqdm(list(remained_relation_to_tail.items())):
    print(rel)
    syn_ents = get_syn_ents(rel, ents)
    relation_to_syn_tail[rel] = syn_ents

with open('./construct_syn_kgqa/relation_to_syn_head.json', 'w') as f:
    json.dump(relation_to_syn_head, f)
    
with open('./construct_syn_kgqa/relation_to_syn_tail.json', 'w') as f:
    json.dump(relation_to_syn_tail, f)
