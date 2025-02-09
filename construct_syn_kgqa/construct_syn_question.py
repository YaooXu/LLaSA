from copy import deepcopy
import json
import random
import re
import sys

from tqdm import tqdm



from utils import load_json, load_jsonl


def find_year(input_string):
    pattern = r"\b\d{4}(?:-\d{2}-\d{2})?\b"
    match = re.search(pattern, input_string)
    if match:
        return True
    else:
        return False


data_filepath = "data/ComplexWebQuestions/ComplexWebQuestions_dev.bound_triples.jsonl"

samples = load_jsonl(data_filepath)

relation_to_syn_head = load_json("construct_syn_kgqa/relation_to_syn_head.json")
relation_to_syn_tail = load_json("construct_syn_kgqa/relation_to_syn_tail.json")

for sample in tqdm(samples):

    ori_ent_to_syn_ent = {}

    question = sample["question"]
    answers = [answer["answer"] for answer in sample["answers"]]

    # print(question)
    # print(answers)

    for triples in sample["all_bound_triples"]:
        # print(triples)
        for triple in triples:
            head, rel, tail = triple

            if not head.isdigit() and not find_year(head) and head not in ori_ent_to_syn_ent:
                if rel in relation_to_syn_head:
                    syn_head = random.choice(relation_to_syn_head[rel])
                    ori_ent_to_syn_ent[head] = syn_head

            if not tail.isdigit() and not find_year(tail) and tail not in ori_ent_to_syn_ent:
                if rel in relation_to_syn_tail:
                    syn_tail = random.choice(relation_to_syn_tail[rel])
                    ori_ent_to_syn_ent[tail] = syn_tail

    print(f"\n{ori_ent_to_syn_ent}\n")

    for ori_ent, syn_ent in ori_ent_to_syn_ent.items():
        if ori_ent in question:
            question = question.replace(ori_ent, syn_ent)
        elif ori_ent.lower() in question:
            question = question.replace(ori_ent.lower(), syn_ent.lower())

    answers = [
        ori_ent_to_syn_ent[answer] if answer in ori_ent_to_syn_ent else answer for answer in answers
    ]

    all_bound_syn_triples = deepcopy(sample["all_bound_triples"])

    for syn_triples in all_bound_syn_triples:
        for triple in syn_triples:
            for i in (0, -1):
                if triple[i] in ori_ent_to_syn_ent:
                    triple[i] = ori_ent_to_syn_ent[triple[i]]

    # print(question)
    # print(answers)

    sample["question"] = question
    sample["answers"] = [{"answer": answer} for answer in answers]
    sample["all_bound_triples"] = all_bound_syn_triples

with open("data/ComplexWebQuestions/ComplexWebQuestions_dev.syn_bound_triples.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
