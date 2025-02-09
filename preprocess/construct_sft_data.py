import sys
sys.path.append('./')

import json
from datasets import load_dataset
import random
from tqdm import tqdm

from utils.tool import get_constructor
from utils.configure import Configure

from transformers import AutoTokenizer, AutoModel

# from dataset.data import TableConverter, GraphConverter
# model_path = "sentence-transformers/all-roberta-large-v1"
# tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
# table_converter, graph_converter = TableConverter(tokenizer), GraphConverter(tokenizer)
# def convert_kg_tups_bidir(kg_tuples):
#     new_kg_tuples = []
#     for tup in kg_tuples:
#         new_kg_tuples.append([f'Node: {tup[0]}', f'Relation: {tup[1]}', f'Node: {tup[2]}'])
#         # new_kg_tuples.append([f'Node: {tup[2]}', f'Inverse Relation: {tup[1]}', f'Node: {tup[0]}'])
#     return new_kg_tuples

# import nltk
# nltk.download('punkt_tab')

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
def truncate_to_max_tokens(text, max_tokens=2048):
    # 使用tokenizer进行token化，确保返回的tokens不会超过 max_tokens
    tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')

    # 将这些tokens转回原始文本
    truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    return truncated_text

prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. "\
        "Write a response that appropriately completes the request.\n\n"\
        "### Instruction:\n\n\n{instruction}\n\n{input}\n\n### Response:\n"

args = Configure.Get('construct_sft_data.cfg')

train_prompt_filepath = 'prompts/instuning_format_spec.json'
with open(train_prompt_filepath, 'r') as f:
    train_prompts_dict = json.load(f)

test_prompt_filepath = 'prompts/instuning_format_spec_eval.json'
with open(test_prompt_filepath, 'r') as f:
    test_prompts_dict = json.load(f)


table_tasks = ["wikitq", "hybridqa", "fetaqa", "tabmwp", "wikisql", "tab_fact", "totto", "kvret", "finqa", "sqa", 'wikitabletext', 'finqa']
schema_tasks = ["spider", "sparc"]
kg_tasks = ['compwebq', 'dart']

def construct_processed_samples(tasks, prompts_dict, is_train, output_path, one_shot=False):
    processed_samples = []
    for task_name in tasks:
        # error_cnt = 0

        task_samples = []

        print(task_name)
        raw_datasets_split = load_dataset(f"preprocess/tasks/{task_name}.py", trust_remote_code=True)
        datasets_split = get_constructor(f"preprocess.seq2seq_construction.{task_name}")(args).to_seq2seq(raw_datasets_split, './data/.cache')

        dataset = datasets_split[0] if is_train else datasets_split[-1]

        if task_name in table_tasks:
            key = 'table'
        elif task_name in schema_tasks:
            key = 'schema_tuples'
        elif task_name in kg_tasks:
            key = 'kg_tuples'

        for sample in tqdm(dataset):
            struct_data = sample[key]

            all_instructions = prompts_dict[task_name]['instruction']
            if isinstance(all_instructions, list):
                instruction = random.choice(all_instructions)
            elif isinstance(all_instructions, str):
                instruction = all_instructions

            input_format = prompts_dict[task_name]['input_format']

            text_in = sample['text_in']
            struct_in = truncate_to_max_tokens(sample['struct_in'])
            seq_out = sample['seq_out']

            if not one_shot:
                input_ = input_format.format(struct_in=struct_in, text_in=text_in)
                if '\n\n\n' in input_:
                    input_ = input_.replace('\n\n\n', '[GRAPH_PAD]')
                else:
                    input_ += '[GRAPH_PAD]'
                
                assert '[GRAPH_PAD]' in input_

                final_input = prompt.format(instruction=instruction, input=input_)
            # else:
            #     input_ = input_format.format(struct_in=struct_in, text_in=text_in)
            #     final_input =  prompt.format(instruction=instruction, input=input_)
            #     final_input = examples[task_name] + '\n\n' + final_input

            label = seq_out

            if is_train:
                task_sample = {
                    'idx': len(processed_samples),
                    'input': final_input,
                    'question': sample['question'],
                    'label': label,
                    key: struct_data,
                    'key': key,
                    'task': task_name,
                    'task_id': len(task_samples),
                }
            else:
                sample.update({
                    'idx': len(processed_samples),
                    'input': final_input,
                    'label': label, 
                    'key': key,
                    'arg_path': f'META_TUNING/{task_name}.cfg',
                    'task': task_name,
                    'task_id': len(task_samples),
                })
                task_sample = sample
                
            task_samples.append(task_sample)

        processed_samples.extend(task_samples)

        #     try:
        #         if processed_sample['key'] == 'table':
        #             graph = table_converter._text2graph(processed_sample['table'], True)
        #         elif processed_sample['key'] == 'schema_tuples':
        #             kg_graph = convert_kg_tups_bidir(processed_sample["schema_tuples"])
        #             graph = graph_converter._kg_tupels2graph(kg_graph, True)
        #         elif processed_sample['key'] == 'kg_tuples':
        #             kg_graph = convert_kg_tups_bidir(processed_sample["kg_tuples"])
        #             graph = graph_converter._kg_tupels2graph(kg_graph, True)
        #     except Exception as e:
        #         table_converter._text2graph(processed_sample['table'], True)
        #         error_cnt += 1
        # print('#error: ', error_cnt, '\n')

        print(processed_samples[-1]['input'])
        print(processed_samples[-1]['label'])
        print(processed_samples[-1]['question'])
        print('\n\n\n')

    print(len(processed_samples))
    with open(output_path, 'w') as f:
        json.dump(processed_samples, f)

# with open('data/processed/1-shot_examples.jsonl', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     examples = [json.loads(line) for line in lines]
#     examples = {list(e.keys())[0]: list(e.values())[0] for e in examples}
# construct_processed_samples(all_tasks, test_prompts_dict, False, 'data/processed/one_shot_test_skginstruct.json')

output_dir = 'data/processed'
all_tasks = ["wikitq", "hybridqa", "fetaqa", "tabmwp", "wikisql", "tab_fact", "totto", "kvret", 'compwebq', 'dart']
construct_processed_samples(all_tasks, train_prompts_dict, True, f'{output_dir}/custom_skginstruct.json')

all_tasks += ['sqa', 'wikitabletext', 'finqa']
construct_processed_samples(all_tasks, test_prompts_dict, False, f'{output_dir}/custom_test_skginstruct.json')