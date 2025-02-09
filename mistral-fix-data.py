import json
import re
from tqdm import tqdm
import ast
from datasets import load_dataset

# prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. "\
#         "Write a response that appropriately completes the request.\n\n"\
#         "### Instruction:\n\n\n{instruction}\n\n{input}\n\n{question}\n\n### Response:\n"

# with open("./data/processed/skginstruct_test_file_7b.json", "r") as f:
#     data = json.load(f)

# # select everything between <</SYS>> and [/INST]
# examples = {}
# new_data = []
# for i, each in tqdm(enumerate(data)):
#     struct_in_idx = each['formatted_input'].find(each['struct_in'][:10])
#     sysend_idx = each['formatted_input'].find("<</SYS>>")
#     instruction = each['formatted_input'][sysend_idx + 8:struct_in_idx].strip()
#     if each['text_in']:
#         text_in_idx = each['formatted_input'].find(each['text_in'])
#         struct_in = each['formatted_input'][struct_in_idx:text_in_idx].strip()
#     else:
#         end  = each['formatted_input'].find("[/INST]")
#         struct_in = each['formatted_input'][struct_in_idx:end].strip()
#     data[i]['formatted_input'] = prompt.format(instruction=instruction, input=struct_in, question=each['text_in'])
#     if each['arg_path'] not in examples:
#         examples[each['arg_path']] = data[i]
#     new_data.append(data[i])

# # write examples formatted inputs into a text file
# printstr = ""
# for k, v in examples.items():
#     printstr += f"### {k}\n\n"
#     printstr += f"{v['formatted_input']}\n\n"
#     printstr += "--------------------------------\n\n"

# with open("data/processed/skginstruct_test_file_mistral_examples.txt", "w") as f:
#     f.write(printstr)


# with open("data/processed/skginstruct_test_file_mistral.json", "w") as f:
#     json.dump(new_data, f)


# # modify tabmwp data format
# with open("data/processed/skginstruct_test_file_mistral.json", "r") as f:
#     samples = json.load(f)
#     tabmwp_samples = [sample for sample in samples if sample['description'] == 'task: tabmwp']

# def convert_to_table(data):
#     header = list(data.keys())
    
#     rows = []
#     num_rows = len(data[header[0]])
#     for i in range(num_rows):
#         row = [data[key][i] for key in header]
#         rows.append(row)
    
#     table = {
#         "header": header,
#         "rows": rows
#     }
    
#     return table

# for sample in tabmwp_samples:
#     sample['table'] = convert_to_table(ast.literal_eval(sample['table_for_pd']))

# print(tabmwp_samples[0]['table'])
# with open("data/processed/skginstruct_test_file_mistral.json", "w") as f:
#     json.dump(samples, f)


# modify dart data format
with open("data/processed/skginstruct_test_file_mistral.json", "r") as f:
    samples = json.load(f)
    dart_samples = [sample for sample in samples if sample['description'] == 'task: dart']

dataset = load_dataset('tasks/dart.py')['test']

text_to_kg_tuples = {}
for sample in dataset:
    text_to_kg_tuples[sample['annotations']['text'][0]] = sample['kg_tuples']

for sample in dart_samples:
    sample['kg_tuples'] = text_to_kg_tuples[sample['seq_out']]

with open("data/processed/skginstruct_test_file_mistral.json", "w") as f:
    json.dump(samples, f)
