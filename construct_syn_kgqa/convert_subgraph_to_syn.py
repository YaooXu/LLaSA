from copy import deepcopy
import json
import random
import re
import sys

from tqdm import tqdm



from utils import convert_kg_tuples_to_str, load_json, load_jsonl


def find_year(input_string):
    pattern = r"\b\d{4}(?:-\d{2}-\d{2})?\b"
    match = re.search(pattern, input_string)
    if match:
        return True
    else:
        return False


prompt = "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.\n<</SYS>>\n\nAnswer the following question with the help of the given list of the knowledge graph triples. knowledge graph triples:\n\n{subgraph}\n\n\nquestion:\n\n{question} [/INST]"

data_filepath = "data/processed/skginstruct_test_file_7b.cwq.json"
ori_output_filepath = "data/processed/skginstruct_test_file_7b.ori_cwq.json"
syn_output_filepath = "data/processed/skginstruct_test_file_7b.syn_cwq.json"


relation_to_syn_head = load_json("construct_syn_kgqa/relation_to_syn_head.json")
relation_to_syn_tail = load_json("construct_syn_kgqa/relation_to_syn_tail.json")

samples = load_json(data_filepath)
ori_samples = []
syn_samples = []
for sample in tqdm(samples):

    ori_ent_to_syn_ent = {}

    question = sample["question"]
    answers = sample["answers"]

    # print(question)
    # print(answers)

    for triple in sample["kg_tuples"]:
        head, rel, tail = triple

        if ".." in rel:
            # some strange relation in the original dataset
            # government.government_office_category.officeholders..government.government_position_held.office_holder
            rel = triple[1] = rel.split("..")[0]

        if not head.isdigit() and not find_year(head) and head not in ori_ent_to_syn_ent:
            if rel in relation_to_syn_head:
                syn_head = random.choice(relation_to_syn_head[rel])
                ori_ent_to_syn_ent[head] = syn_head

        if not tail.isdigit() and not find_year(tail) and tail not in ori_ent_to_syn_ent:
            if rel in relation_to_syn_tail:
                syn_tail = random.choice(relation_to_syn_tail[rel])
                ori_ent_to_syn_ent[tail] = syn_tail

    new_question = question
    for ori_ent, syn_ent in ori_ent_to_syn_ent.items():
        if ori_ent in question:
            new_question = new_question.replace(ori_ent, syn_ent)
        elif ori_ent.lower() in question:
            new_question = new_question.replace(ori_ent.lower(), syn_ent.lower())

    new_answers = [
        ori_ent_to_syn_ent[answer] if answer in ori_ent_to_syn_ent else answer for answer in answers
    ]

    if new_question == question:
        continue

    ori_samples.append(sample)

    syn_sample = deepcopy(sample)

    all_syn_triples = syn_sample["kg_tuples"]

    for triple in all_syn_triples:
        for i in (0, -1):
            if triple[i] in ori_ent_to_syn_ent:
                triple[i] = ori_ent_to_syn_ent[triple[i]]

    # print(question)
    # print(answers)
    syn_sample["question"] = new_question
    syn_sample["answers"] = new_answers
    syn_sample["seq_out"] = ", ".join(syn_sample["answers"])
    syn_sample["kg_tuples"] = all_syn_triples
    syn_sample["struct_in"] = convert_kg_tuples_to_str(syn_sample["kg_tuples"])

    new_formatted_input = prompt.format(
        subgraph=syn_sample["struct_in"], question=syn_sample["question"]
    )
    syn_sample["formatted_input"] = new_formatted_input

    syn_samples.append(syn_sample)

    # print(f'{question}\n{answers}\n')
    # print(f"{len(ori_ent_to_syn_ent)}\n{ori_ent_to_syn_ent}\n")
    # print(f'{new_question}\n{new_answers}\n')

assert len(ori_samples) == len(syn_samples)
print(len(syn_samples))

with open(ori_output_filepath, "w") as f:
    f.write(json.dumps(ori_samples) + "\n")

with open(syn_output_filepath, "w") as f:
    f.write(json.dumps(syn_samples) + "\n")
