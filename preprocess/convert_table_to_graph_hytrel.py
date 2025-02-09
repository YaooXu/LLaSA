import sys
sys.path.append('./')

from copy import deepcopy
import random
import time
import zipfile
import sys

import numpy as np


import os

import json
import pickle
import re

import pandas as pd
import torch
from tqdm import tqdm
from utils.utils import df_to_jsonl, load_json, write_jsonl

from collections import defaultdict, deque
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool
from datasets import load_dataset, Dataset, DownloadMode
import torch.nn.functional as F

from dataset.data import TableConverter, GraphConverter
from utils.sentence_transformer import get_sentence_embeds
from multiprocessing import Pool

# constants
CAP_TAG = "<caption>"
HEADER_TAG = "<header>"
ROW_TAG = "<row>"

MISSING_CAP_TAG = "[TAB]"
MISSING_CELL_TAG = "[CELL]"
MISSING_HEADER_TAG = "[HEAD]"

import multiprocessing

# 创建一个锁字典来存储不同 zip 文件路径对应的锁
lock_dict = defaultdict(multiprocessing.Lock)

def convert_kg_tups_bidir(kg_tuples):
    new_kg_tuples = []
    for tup in kg_tuples:
        new_kg_tuples.append([f'Node: {tup[0]}', f'Relation: {tup[1]}', f'Node: {tup[2]}'])
        new_kg_tuples.append([f'Node: {tup[2]}', f'Inverse Relation: {tup[1]}', f'Node: {tup[0]}'])
    return new_kg_tuples


def obtain_samples(process_idx, samples, cache_dir, cache_graphs, pretraining, model_path, tokenizer):
    def get_zip_file_and_name(sample):
        zip_file_path = f"{cache_dir}/{sample['task']}/{sample['task_id'] // 2000}.zip"
        graph_name = f"graph_{sample['task_id']}.pkl"
        return zip_file_path, graph_name

    if cache_graphs:
        llm = AutoModel.from_pretrained(
            model_path,
        ).to(f"cuda:{process_idx}")
        llm.eval()
        
    table_converter, graph_converter = TableConverter(tokenizer), GraphConverter(tokenizer)

    new_samples = []
    t1 = time.time()
    for n, sample in enumerate(samples):
        if (n + 1) % 500 == 0:
            t2 = time.time()
            print(f"{process_idx}: {n / len(samples)}, {t2 - t1}")
            t1 = t2

        try:
            if sample['key'] == 'table':
                graph = table_converter._text2graph(sample['table'], True)
            elif sample['key'] == 'schema_tuples':
                kg_graph = convert_kg_tups_bidir(sample["schema_tuples"])
                graph = graph_converter._kg_tupels2graph(kg_graph, True)
            elif sample['key'] == 'kg_tuples':
                kg_graph = convert_kg_tups_bidir(sample["kg_tuples"])
                graph = graph_converter._kg_tupels2graph(kg_graph, True)

            sample["graph"] = graph
            if graph:
                sample['num_nodes'] = graph['num_nodes'] + graph['num_hyperedges']

                zip_file_path, graph_name = get_zip_file_and_name(sample)
                if cache_graphs:
                    parent_dir = os.path.dirname(zip_file_path)
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)

                    graph = sample.pop("graph")

                    embedding_s = get_sentence_embeds(llm, tokenizer, torch.LongTensor(graph["x_s"]).to(llm.device), batch_size=2048)
                    graph["x_s"] = embedding_s.detach().cpu().float().numpy()

                    embedding_t = get_sentence_embeds(llm, tokenizer, torch.LongTensor(graph["x_t"]).to(llm.device), batch_size=2048)
                    graph["x_t"] = embedding_t.detach().cpu().float().numpy()

                    del embedding_s, embedding_t

                    lock = lock_dict[zip_file_path]
                    with lock:
                        with zipfile.ZipFile(zip_file_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                            sample["graph_path"] = (zip_file_path, graph_name)
                            
                            serialized_dict = pickle.dumps(graph)
                            zipf.writestr(graph_name, serialized_dict)
                else:

                    if not pretraining:
                        graph = sample.pop("graph")

                        sample["graph_path"] = (zip_file_path, graph_name)
                    else:
                        # graph in remained in pretraining
                        pass

                new_samples.append(sample)
        except Exception as e:
            print(e)
            continue

    return new_samples

if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    set_start_method("spawn", force=True)
    
    n_process = torch.cuda.device_count() # num GPU if cache
    shuffle = False
    model_path = "sentence-transformers/all-roberta-large-v1"
    
    # pretraining = True
    # output_dir = f"./data/hytrel/llm_based_gnn_pretraining"
    
    pretraining = False
    output_dir = f"data/all-table-kg-schema-tasks"
    os.makedirs(output_dir, exist_ok=True)

    cache_graphs = False
    cache_dir = f"data/all-table-kg-schema-tasks"

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for path in (
        # 'data/processed/llm_based_gnn_pretraining.pq',
        "data/processed/custom_skginstruct.json",
        "data/processed/custom_test_skginstruct.json",
    ):
        all_samples = load_json(path=path) if path.endswith('json') else load_dataset("parquet", data_files=path, cache_dir='./data/.cache')['train']
        num_samples = len(all_samples)
        print(num_samples)

        cur_cache_dir = os.path.join(cache_dir, "test" if "test" in path else "train")
        os.makedirs(cur_cache_dir, exist_ok=True)

        with Pool(processes=n_process) as pool:
            num_samples_in_chunk = num_samples // n_process + 1
            jobs = []
            st = 0
            for i in range(n_process):
                ed = st + num_samples_in_chunk
                ed = min(ed, num_samples)
                jobs.append([i, all_samples[st:ed], cur_cache_dir, cache_graphs, pretraining ,model_path, tokenizer])
                st = ed

            results = pool.starmap(obtain_samples, jobs)

        all_samples = []
        for samples in results:
            all_samples.extend(samples)

        print(len(all_samples))

        if "test" in path:
            df = pd.DataFrame(all_samples)

            remain_keys = ["label", "input", "question", "graph_path", "task", "num_nodes"]
            sub_df = df[remain_keys]

            dataset = Dataset.from_pandas(sub_df)
            dataset.to_parquet(f"{output_dir}/test.pq")
            dataset.to_parquet(f"{output_dir}/val.pq")

            write_jsonl(f"{output_dir}/ori_test.jsonl", all_samples)
            write_jsonl(f"{output_dir}/ori_val.jsonl", all_samples)

        else:
            print(all_samples[0])
            df = pd.DataFrame(all_samples)
            dataset = Dataset.from_pandas(df)
            dataset.to_parquet(f"{output_dir}/train.pq")

            # from datasets import load_dataset
            # load_dataset("parquet", data_files=f"{output_dir}/train.pq")['train']

        print("done")