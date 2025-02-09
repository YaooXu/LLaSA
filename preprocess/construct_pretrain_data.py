import sys

import datasets

sys.path.append("./")

from datasets import load_dataset
from dataset.data import TableConverter, GraphConverter
from transformers import AutoTokenizer
import random
import numpy as np
import networkx as nx
import torch
from datasets import concatenate_datasets


def random_cell_index(num_rows, num_cols, ignore_first_col=False):
    row_index = random.randint(0, num_rows - 1)

    col_st_idx = 1 if ignore_first_col else 0
    col_index = random.randint(col_st_idx, num_cols - 1)

    return (row_index, col_index)


def sample_ignoring_element(lst, ignore_element):
    filtered_list = [elem for elem in lst if elem != ignore_element]

    sampled_element = random.choice(filtered_list)

    return sampled_element


def construct_table_pretraining_questions(table, k=20):
    # table: headers : [], rows: [[...], [...], ]
    num_rows = len(table["rows"])
    num_cols = len(table["header"])

    def construct_template1():
        template = 'What\'s the column name of "{node_name}".'

        index = random_cell_index(num_rows, num_cols)
        node_name = table["rows"][index[0]][index[1]]
        col_name = table["header"][index[1]]

        if node_name == "" or col_name == "":
            return []

        question = template.format(node_name=node_name)

        answer = col_name

        return [(question, answer)]

    def construct_template2():
        template = (
            'In the row where the value of "{first_col_name}" is "{row_value}", what is the corresponding value of "{col_name}?"'
        )

        index = random_cell_index(num_rows, num_cols, ignore_first_col=True)
        node_name = table["rows"][index[0]][index[1]]
        col_name = table["header"][index[1]]
        first_col_name = table["header"][0]

        if node_name == "" or col_name == "" or first_col_name == "":
            return []

        row_value = table["rows"][index[0]][0]

        question = template.format(col_name=col_name, first_col_name=first_col_name, row_value=row_value)
        answer = node_name

        return [(question, answer)]

    def construct_template3():
        qa_pairs = []

        template = 'Are "{node_name1}" and "{node_name2}" in the same row?'
        index1 = random_cell_index(num_rows, num_cols)
        node_name1 = table["rows"][index1[0]][index1[1]]

        # same row
        col_idx2 = sample_ignoring_element(list(range(num_cols)), index1[1])
        index2 = (index1[0], col_idx2)
        node_name2 = table["rows"][index2[0]][index2[1]]
        question = template.format(node_name1=node_name1, node_name2=node_name2)
        if not (node_name1 == "" or node_name2 == ""):
            qa_pairs.append((question, "True"))

        # different row
        row_idx3 = sample_ignoring_element(list(range(num_rows)), index2[0])
        index3 = (row_idx3, index2[1])
        node_name3 = table["rows"][index3[0]][index3[1]]
        question = template.format(node_name1=node_name1, node_name2=node_name3)
        if not (node_name1 == "" or node_name3 == ""):
            qa_pairs.append((question, "False"))

        return qa_pairs

    functions = [construct_template1, construct_template2, construct_template3]
    all_qa_pairs = set()

    for _ in range(k):
        function = random.choice(functions)
        for qa_pair in function():
            all_qa_pairs.add(qa_pair)
    return list(all_qa_pairs)


def preprocess_table(samples, k=20):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    table_converter = TableConverter(tokenizer)

    dataset = []
    for table in samples["table"]:
        graph = table_converter._text2graph(table, return_dict=True)

        qa_pairs = construct_table_pretraining_questions(table, k=k)

        if len(qa_pairs) >= 10:
            dataset.append({
                "graph": graph,
                "question": [question for question, answer in qa_pairs],
                "label": [answer for question, answer in qa_pairs],
            })
            
    results = {key: [d[key] for d in dataset] for key in dataset[0]}
    return results


if __name__ == "__main__":
    model_path = "sentence-transformers/all-roberta-large-v1"
    num_proc = 64

    file_path = './TaBERT/data/preprocessed_data/data.pq'
    dataset = load_dataset("parquet", data_files=file_path)['train']
    
    # remove_columns=["table"] is necessary
    dataset = dataset.map(preprocess_table, batched=True, num_proc=num_proc, load_from_cache_file=False, remove_columns=["table"])
    print(len(dataset))
    
    output_dir = 'data/pretraining_25M_tables'
    dataset.to_parquet(f'{output_dir}/train.pq')
    dataset.select(range(100)).to_parquet(f'{output_dir}/val.pq')
    dataset.select(range(100)).to_parquet(f'{output_dir}/test.pq')